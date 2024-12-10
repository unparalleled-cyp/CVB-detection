# -*- coding: utf-8 -*-
"""
@File: dataset_bert.py 
@Time : 2024/11/29 11:38

@dec: 替换为BertDataset
（1）文本处理：改用 BERT 的分词器（如 transformers 中的 BertTokenizer）处理文本。
（2）特征输入：生成 [input_ids, attention_mask, token_type_ids] 等 BERT 输入格式。
（3）数据格式：用 PyTorch Dataset 接口（继承 torch.utils.data.Dataset）重写数据加载逻辑。
"""

import os
import torch

from torch.utils.data import DataLoader, Dataset
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer

from utils.data_process import DataProcess
from utils.clean_data import clean_string, cut_sentence
# from data_process import DataProcess
# from clean_data import clean_string, cut_sentence

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
    对文件名进行分裂，分裂成训练集和验证集，要设置seed
    :param fileholder: 数据集文件夹
    :param seed: seed种子
    :param mode: 模式，是训练模式还是验证模式
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

class BertDataset(object):
    """
    Bert数据集
    1. 使用BERT Tokenizer对文本进行编码；
    2. 支持训练集和验证集的划分；
    3. 返回 BERT 所需的 input_ids、attention_mask、token_type_ids 和 labels。
    """
    def __init__(self, fileholder, seed, mode, vocab_filename, shuffle, use_cuda, device, max_length=128, bert_model="../../bert-base-chinese"):
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
        # 初始化 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_length = max_length
        self.use_cuda = use_cuda
        self.device = device
#         self.bert = BertModel.from_pretrained(bert_model)
        
        # 获得文件夹中所有文件有标签的内容
        self.label_map = {'0': "-1", '1': "0", "2": "0", '3': "1", "4": "2"}   # 标签映射规则
        filenames = split_filenames(fileholder=fileholder, seed=seed, mode=mode) # 根据模式划分文件
        data_processor = DataProcess(mode, True) # 初始化数据处理器
        datasets = data_processor.run(filenames, 15) # 加载数据，包含内容、标签等

        # process data
        self.contents = np.array([d["content"] for d in datasets]) # 文本内容
        self.labels = np.array([d["label"] for d in datasets]) # 标签
        self.adjs = np.array([d["adj"] for d in datasets]) # 邻接矩阵
        self.user_infos = np.array([d["user_infos"] for d in datasets])
        self.cids = np.array([d["cid"] for d in datasets]) # 用户 ID 或其他标识

        self.char_vocabs = self.read_vocab_filename(vocab_filename)

        # 配置参数
        self.shuffle = shuffle
        self.start_idx = 0
    def read_vocab_filename(self, vocab_filename):
        """
        从vocab字典中读取字，变成字典格式
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
        # self.contents, self.labels, self.adjs, self.user_infos, self.cids = \
        #     self.contents[ids], self.labels[ids], self.adjs[ids], self.user_infos[ids], self.cids[ids]
        self.contents, self.labels, self.adjs, self.user_infos = (
            self.contents[ids],
            self.labels[ids],
            self.adjs[ids],
            self.user_infos[ids],
        )


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
        return res

    def _batchify(self, contents, labels, adj, user_infos):
        """
        对单条数据进行编码并返回 BERT 所需格式
        :param content: 文本内容
        :param label: 标签
        :param adj: 邻接矩阵
        :return: (input_ids, attention_mask, token_type_ids, adj, label)
        """

        input_ids, attention_mask = self._deal_text(contents) #, token_type_ids
        # Convert labels to tensors
#         labels = torch.tensor(labels, dtype=torch.long)
        

        user_infos = self._deal_user_infos(user_infos)
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        
#         print(f"token_type_ids: {token_type_ids}")
#         print(f"type(token_type_ids): {type(token_type_ids)}")
        
#         token_type_ids = torch.stack(token_type_ids)
#         token_type_ids = token_type_ids.squeeze(dim=-1)
#         print(f"token_type_ids: {token_type_ids}")
#         print(f"type(token_type_ids): {type(token_type_ids)}")
        labels = torch.tensor(labels, dtype=torch.long)
                                                             
        adj = torch.tensor(adj, dtype=torch.double) # float吗？？？

        if self.use_cuda:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
#             token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            adj = adj.to(self.device)
            for key in user_infos:
                user_infos[key] = user_infos[key].to(self.device)
        
        return (input_ids, attention_mask), adj, labels, user_infos #, token_type_ids

    def _deal_text(self, contents):
        input_ids_list = []
        attention_masks_list = []
#         token_type_ids_list = []

        for text in contents:
            # 使用 tokenizer 进行编码
            encoding = self.tokenizer.encode_plus(
                clean_string(text), # text
                add_special_tokens=True,  # 添加特殊标记，例如 [CLS], [SEP]
                max_length=self.max_length,  # 最大序列长度
                padding="max_length",  # 按最大长度填充
                truncation=True,  # 截断超出 max_length 的部分
                return_tensors="pt"  # 返回 PyTorch 张量
            )
            
            # 获取编码结果
            input_ids = encoding["input_ids"].squeeze(0)  # 移除批次维度
            attention_mask = encoding["attention_mask"].squeeze(0)  # 同上
            
            # 有些模型不需要 token_type_ids，可选默认值
#             token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))

            # 收集编码结果
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
#             token_type_ids_list.append(token_type_ids)

        return input_ids_list, attention_masks_list#, token_type_ids_list
    
    def _deal_user_text(self, contents):
        """
        对文本序列进行处理：
        1、先将字符变成索引
        2、在进行padding
        3、最后获得mask
        """
        # 获得contents所对应的索引
        ids = []
        for text in contents:
            text_id = [self.char_vocabs[c] if c in self.char_vocabs else 1 for c in cut_sentence(clean_string(text))]
            ids.append(text_id)

        # 通过ids获得它所对应的batch_ids、batch_len、masks
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
        _, _, outputs["verify_info_coupon"], outputs["verify_info_coupon_mask"] = self._deal_user_text(verify_info_coupon)
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
    fileholder = "../../dataset/data_bak/1.5-s125/"
    dataLoader = BertDataset(fileholder, seed=0, mode="train", vocab_filename="../../embeddings/vocab.txt", shuffle=False, use_cuda=False, device=None, bert_model="../../bert-base-chinese")
    
    labels_dict = {}  # 用于统计标签的分布
    label_types = set()  # 用于记录标签的所有数据类型

    for i, data in enumerate(dataLoader):
        labels = data[-2]  # 假设 data[-2] 是 labels
        # 检查每个 label 的类型和值
        for label in labels:
            # 添加数据类型到集合
            label_types.add(type(label))

            # 如果是 Tensor，需要转换为 Python 的数值类型
            if isinstance(label, torch.Tensor):
                label = label.item()

            # 统计标签的值
            if label not in labels_dict:
                labels_dict[label] = 0
            labels_dict[label] += 1

    # 输出标签数据类型和分布
    print("Label types:", label_types)  # 输出所有标签的类型
    print("Label values distribution:", labels_dict)  # 输出标签的分布

#     labels_dict = {}
#     label_types = set()
#     for i, data in enumerate(dataLoader):
#         l = data[-1]
#         for o in l:
#             o = o.item()
#             if o not in labels_dict:
#                 labels_dict[o] = 0
#             labels_dict[o] += 1
#     print(labels_dict)