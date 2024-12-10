# -*- coding: utf-8 -*-
"""
@File: user_info_embedding.py 
@Time : 2022/5/27 10:52

@dec: 
"""


import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F


class SparseEmbedding(nn.Module):
    """
    对离散的类别进行编码：
        1、gender  未知、男、女
        2、focus_num\fans_num\blogs_num  当然先要进行分割，分成0-10,10-50,...
        3、verify  微博个人认证，微博官方认证，其他的都认为是无认证
        4、vip  0~7
        5、学历  没填学历、匹配不上学历，真实学历
        6、place  其他、具体省份
    """
    def __init__(self, sparse_num, char_dim, hidden_dim, pretrained_char_emb_filename):
        super(SparseEmbedding, self).__init__()
        self.char_embedding_layer = nn.Embedding(sparse_num, char_dim).to(torch.double)
        self.char_embedding_layer.weight.data.copy_(
            torch.from_numpy(self.random_embedding(sparse_num, char_dim)))
        self.dense = nn.Linear(char_dim, hidden_dim).to(torch.double)

    def forward(self, user_sparse_idx):
        outputs = self.char_embedding_layer(user_sparse_idx)
        outputs = self.dense(outputs)
        return outputs

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def read_pretrained_char_emb(self, filename):
        """
        从filename读取权重参数
        :param filename:
        :return:
        """
        weights = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    nums = list(map(float, line.split("\t")[1:]))
                    assert len(nums) == 50
                    weights.append(nums)
        return np.array(weights)

    def reset_parameters(self):
        nn.init.orthogonal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)


class TextEmbedding(nn.Module):
    """
    对文本进行编码：
        1、verify_info
        2、coupon
        用句号把他们连接起来
    """
    def __init__(self, char_size, char_dim, hidden_dim, lstm_layer, pretrained_char_emb_filename):
        super(TextEmbedding, self).__init__()
        self.char_embedding_layer = nn.Embedding(char_size, char_dim).to(torch.double)
        if pretrained_char_emb_filename is not None:
            # 先读取文件
            pretrained_char_emb = self.read_pretrained_char_emb(pretrained_char_emb_filename)
            self.char_embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_char_emb))
        else:
            self.char_embedding_layer.weight.data.copy_(
                torch.from_numpy(self.random_embedding(char_size, char_dim)))
        self.lstm = nn.LSTM(char_dim, hidden_dim//2, num_layers=lstm_layer, batch_first=True, bidirectional=True).to(torch.double)
        self.reset_parameters()

    def forward(self, user_text_idx, masks):
        outputs = self.char_embedding_layer(user_text_idx)
        outputs = self.lstm(outputs)[0]
        outputs = outputs - (1 - masks.unsqueeze(-1)) * 1e8
        outputs = torch.max(outputs, dim=-2)[0]
        return outputs

    def reset_parameters(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)

    def read_pretrained_char_emb(self, filename):
        """
        从filename读取权重参数
        :param filename:
        :return:
        """
        weights = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    nums = list(map(float, line.split("\t")[1:]))
                    assert len(nums) == 50
                    weights.append(nums)
        return np.array(weights)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb


class UserInfoEmbedding(nn.Module):
    def __init__(self,
                 sparse_num_dict: dict,
                 char_size: int,
                 char_dim: int,
                 hidden_dim: int,
                 lstm_layer: int,
                 dropout_rate: float,
                 pretrained_char_emb_filename: str):
        """
        :param sparse_num_dict: 为字典格式
                1、gender  未知、男、女
                2、focus_num\fans_num\blogs_num  当然先要进行分割，分成0-10,10-50,...
                3、verify  微博个人认证，微博官方认证，其他的都认为是无认证
                4、vip  0~7
                5、edu 学历  没填学历、匹配不上学历，真实学历
                6、place  其他、具体省份
        :param char_size:
        :param char_dim:
        :param hidden_dim:
        :param lstm_layer:
        :param dropout_rate:
        :param pretrained_char_emb_filename:
        """
        super(UserInfoEmbedding, self).__init__()
        self.gender_embedding_module, self.focus_num_embedding_module, self.fans_num_embedding_module,\
            self.blogs_num_embedding_module, self.verify_embedding_module, self.vip_embedding_module,\
            self.edu_embedding_module, self.place_embedding_module = [
            SparseEmbedding(sparse_num_dict[typ], char_dim, hidden_dim, pretrained_char_emb_filename)
            for typ in ["gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place"]
        ]
        self.text_embedding_module = TextEmbedding(
            char_size=char_size,
            char_dim=char_dim,
            hidden_dim=hidden_dim,
            lstm_layer=lstm_layer,
            pretrained_char_emb_filename=pretrained_char_emb_filename)
        self.dropout = nn.Dropout(dropout_rate)
        self.final_dense = nn.Linear(hidden_dim, hidden_dim).to(torch.double)
        self.reset_parameters()

    def forward(self, user_infos):
        gender, focus_num, fans_num, blogs_num, verify, vip, edu, place = [
            self._forward_sparse_embedding(user_infos[typ], typ)
            for typ in ["gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place"]
        ]
        verify_info_coupon, verify_info_coupon_mask = user_infos["verify_info_coupon"], user_infos["verify_info_coupon_mask"]
        verify_info_coupon = self.text_embedding_module(verify_info_coupon, verify_info_coupon_mask)
        outputs = torch.mean(torch.stack([gender, focus_num, fans_num, blogs_num, verify, vip, edu, place, verify_info_coupon], dim=-1), dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.final_dense(outputs)
        return outputs

    def _forward_sparse_embedding(self, inputs, typ):
        """
        对inputs进行embedding，类型是typ
        """
        module = getattr(self, typ + "_embedding_module")
        outputs = module(inputs)
        return outputs

    def reset_parameters(self):
        nn.init.orthogonal_(self.final_dense.weight)
        nn.init.constant_(self.final_dense.bias, 0)
