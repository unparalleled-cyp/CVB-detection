# -*- coding: utf-8 -*-
"""
@File: dataset.py 
@Time : 2023/5/5 14:44

@dec: 
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from utils.data_process import DataProcess
from utils.clean_data import clean_string, cut_sentence


def read_excel(filename):
    df = pd.read_excel(filename, sheet_name=0)
    return df


def read_dir(fileholder):
    fileholder = os.path.join(fileholder, "*.xlsx")
    filenames = glob(fileholder)
    filenames = [f for f in filenames if "dataAll.xlsx" not in f]
    return filenames


def split_filenames(fileholder, seed, mode):
    """
    train/ val set seed
    :param fileholder: dataset file
    :param rate: val ratio
    :param seed: seed
    :param mode: train/predict/test
    :return:
    """
    if mode == 'train':
        filenames = read_dir(fileholder + 'train')
    elif mode == 'val':
        filenames = read_dir(fileholder + 'dev')
    elif mode == "predict":
        f1 = read_dir(fileholder + 'train')
        f2 = read_dir(fileholder + 'dev')
        filenames = f1 + f2
    else:
        raise Exception("请输入正确的mode, 必须为train、val、predict, 但是你输入的是：%s" % mode)

    filenames.sort()
    np.random.seed(seed)
    filenames = np.array(filenames)
    if mode != "predict":
        np.random.shuffle(filenames)
    return filenames
    # val_filenames = filenames[: int(len(filenames) * rate)]
    # train_filenames = filenames[int(len(filenames) * rate) : ]
    # if mode == "train":
    #     return train_filenames
    # else:
    #     return val_filenames


class BiLSTMDataset(object):
    """
    BiLSTM
    """
    def __init__(self, fileholder, seed, mode, vocab_filename, shuffle, use_cuda, device):
        """
        initial
        :param fileholder: name
        :param dev_rate: val ratio
        :param seed: random seed
        :param mode: train，val
        :param vocab_filename: 
        :param batch_size:
        :param shuffle: 
        :param use_cuda: 
        :param device: 
        """
        
        self.label_map = {'0': "-1", '1': "0", "2": "0", '3': "1", "4": "2"}  
        filenames = split_filenames(fileholder=fileholder, seed=seed, mode=mode)
        data_processor = DataProcess(mode, True) # 改True
        datasets = data_processor.run(filenames, 15)
        self.contents = np.array([str(d["content"]) for d in datasets])
        self.labels = np.array([str(d["label"]) for d in datasets])
        self.adjs = np.array([str(d["adj"]) for d in datasets])
        self.user_infos = np.array([str(d["user_infos"]) for d in datasets])
        self.cids = np.array([str(d["cid"]) for d in datasets])

        self.char_vocabs = self.read_vocab_filename(vocab_filename)

        self.shuffle = shuffle
        self.use_cuda = use_cuda
        self.device = device

        self.start_idx = 0

    def read_vocab_filename(self, vocab_filename):
        """
        :param vocab_filename:
        :return:
        """
        vocabs = {"UNK": 1, "PAD": 0}
        with open(vocab_filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and line in vocabs:
                    raise Exception("字[%i]已经在字典中" % line)
                vocabs[line] = len(vocabs)
        return vocabs

    def _shuffle(self):
        """
        shuffle
        :return:
        """
        ids = np.arange(len(self.contents))
        np.random.shuffle(ids)
        self.contents, self.labels, self.adjs, self.user_infos, self.cids = \
            self.contents[ids], self.labels[ids], self.adjs[ids], self.user_infos[ids], self.cids[ids]

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx == 0 and self.shuffle: # KJTODO
           self._shuffle()

        if self.start_idx >= len(self.contents):
            self.start_idx = 0
            raise StopIteration

        res = self._batchify(self.contents[self.start_idx],
                             self.labels[self.start_idx],
                             self.adjs[self.start_idx],
                             self.user_infos[self.start_idx])
        cids = self.cids[self.start_idx]
        self.start_idx += 1
        return (cids,) + res

    def _batchify(self, contents, labels, adj, user_infos):
        """
        batch_ids  padding
             batch_len,  
             masks
        sort
        :param contents: 
        :param labels: 
        :param adj: 
        :return:
        """
        max_length, batch_len, batch_char, mask = self._deal_text(contents)
        user_infos = self._deal_user_infos(user_infos)

        batch_char, batch_len, mask, labels = torch.tensor(batch_char), \
                   torch.tensor(batch_len), torch.tensor(mask, dtype=torch.double), \
                   torch.tensor(labels, dtype=torch.long)

        adj = torch.tensor(adj, dtype=torch.double)
        if self.use_cuda:
            res = (batch_char.to(self.device), batch_len.cpu(), mask.to(self.device)), adj.to(self.device), labels.to(self.device), user_infos
            return res
        return (batch_char, batch_len, mask), adj, labels, user_infos

    def _deal_text(self, contents):
        ids = []
        for text in contents:
            text_id = [self.char_vocabs[c] if c in self.char_vocabs else 1 for c in cut_sentence(clean_string(text))]
            ids.append(text_id)

        # batch_ids、batch_len、masks
        batch_len = np.array([len(i) for i in ids])
        max_length = int(np.max(batch_len))
        batch_char = np.zeros((len(ids), max_length), dtype=int)
        for i in range(len(ids)):
            batch_char[i][:batch_len[i]] = np.array(ids[i])
        mask = np.array(batch_char > 0)
        return max_length, batch_len, batch_char, mask

    def _deal_user_infos(self, user_info):

        shuxing = ["gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place", "verify_info_coupon"]
        outputs = dict()
        for typ in shuxing[:-1]:
            if self.use_cuda:
                outputs[typ] = self._deal_sparse(user_info, typ).to(self.device)
            else:
                outputs[typ] = self._deal_sparse(user_info, typ)
        verify_info_coupon = [info["verify_info_coupon"] for info in user_info]
        _, _, outputs["verify_info_coupon"], outputs["verify_info_coupon_mask"] = self._deal_text(verify_info_coupon)
        if self.use_cuda:
            outputs["verify_info_coupon"] = torch.tensor(outputs["verify_info_coupon"]).to(self.device)
        else:
            outputs["verify_info_coupon"] = torch.tensor(outputs["verify_info_coupon"])
        if self.use_cuda:
            outputs["verify_info_coupon_mask"] = torch.tensor(outputs["verify_info_coupon_mask"], dtype=torch.double).to(self.device)
        else:
            outputs["verify_info_coupon_mask"] = torch.tensor(outputs["verify_info_coupon_mask"], dtype=torch.double)
        return outputs

    def _deal_sparse(self, user_infos, typ):
        """
        "gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place"
        """
        outputs = []
        for info in user_infos:
            outputs.append(info[typ])
        return torch.tensor(outputs)



if __name__ == "__main__":
    fileholder = "../../data2/"
    dataLoader = BiLSTMDataset(fileholder, 0.0, 1, "train", vocab_filename="../vocab.txt", shuffle=False, use_cuda=False, device=None)

    labels_dict = {}
    for i, data in enumerate(dataLoader):
        l = data[-1]
        for o in l:
            o = o.item()
            if o not in labels_dict:
                labels_dict[o] = 0
            labels_dict[o] += 1
    print(labels_dict)
