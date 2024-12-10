# -*- coding: utf-8 -*-
"""
@File: BiLSTM_Graph.py 
@Time : 2022/5/15 0:55

@dec: BiLSTM-GRAPH网络
"""


import torch
from torch import nn

from model.bilstm_maxpool import BLSTM_MAXPOOL
from model.gatlayer import GAT
from model.user_info_embedding import UserInfoEmbedding


class BiLSTM_Graph(nn.Module):
    """
    构建方法：一次只处理一张图
    1、inputs进来之后先进行BiLSTM，获得inputs对应编码的向量lstm_outputs
    2、将lstm_outputs和adj一起输入到GAT中，注意lstm_outputs和adj的第0维必须要是1
    """
    def __init__(self,
                 char_emb_dim,
                 lstm_hidden_dim,
                 lstm_layer,
                 char_alphabet_size,
                 pretrained_char_emb_filename,
                 dropout_rate,
                 gat_dropout_rate,
                 label_size,
                 gat_hidden_dim=128,
                 alpha=0.1,
                 gat_nheads=4,
                 gat_layer=2):
        super(BiLSTM_Graph, self).__init__()
        self.char_emb_dim = char_emb_dim,
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layer = lstm_layer
        self.char_alphabet_size = char_alphabet_size
        self.pretrained_char_emb_filename = pretrained_char_emb_filename,
        self.dropout_rate = dropout_rate
        self.gat_dropout_rate = gat_dropout_rate
        self.label_size = label_size
        self.gat_hidden_dim = gat_hidden_dim
        self.alpha = alpha
        self.gat_nheads = gat_nheads
        self.gat_layer = gat_layer

        self.BiLSTM = BLSTM_MAXPOOL(
            char_emb_dim = char_emb_dim,
            hidden_dim = lstm_hidden_dim,
            lstm_layer = lstm_layer,
            char_alphabet_size = char_alphabet_size,
            pretrained_char_emb_filename = pretrained_char_emb_filename,
            dropout_rate = dropout_rate,
            label_size = label_size
        )
        rate = 1/8
        self.GAT = GAT(
            nfeat=int(lstm_hidden_dim*(1+rate)),
            nhid=gat_hidden_dim,
            nclass=label_size,
            dropout=gat_dropout_rate,
            alpha=alpha,
            nheads=gat_nheads,
            layer=gat_layer)
        sparse_num_dict = {
            "gender": 3,
            "focus_num": 10,
            "fans_num": 14,
            "blogs_num": 14,
            "verify": 3,
            "vip": 8,
            "edu": 7,
            "place": 37
        }
        self.UserInfo = UserInfoEmbedding(
             sparse_num_dict = sparse_num_dict,
             char_size = char_alphabet_size,
             char_dim = char_emb_dim,
             hidden_dim = int(lstm_hidden_dim*rate),
             lstm_layer = lstm_layer,
             dropout_rate = dropout_rate,
             pretrained_char_emb_filename = pretrained_char_emb_filename)
        self.Dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(int(lstm_hidden_dim*(2+rate)), label_size).to(torch.double)
        # self.w1 = nn.Parameter(torch.tensor(1.0, dtype=torch.double))
        # self.w2 = nn.Parameter(torch.tensor(1.0, dtype=torch.double))

    def forward(self, batch_char, batch_len, mask, adj, user_infos):
        # LSTM
        lstm_outputs = self.BiLSTM(batch_char, batch_len, mask) # kjtodo
        # User_info
        user_info_outputs = self.UserInfo(user_infos)
        lstm_outputs = torch.cat([lstm_outputs, user_info_outputs], dim=-1)  # [node_num, dim]
        # lstm_outputs = torch.mul(lstm_outputs_o, user_info_outputs)
        # GAT
        # lstm_outputs_o = lstm_outputs_o.unsqueeze(0)  # [bs, node_num ,dim]  bs=1
        lstm_outputs = lstm_outputs.unsqueeze(0)  # [bs, node_num ,dim]  bs=1
        # user_info_outputs = user_info_outputs.unsqueeze(0)  # [bs, node_num ,dim]  bs=1
        adj = adj.unsqueeze(0)  # [1, node_num, node_num]
        gat_outputs = self.GAT(lstm_outputs, adj)
        
        # outputs = self.w1 * lstm_outputs + self.w2 * gat_outputs
        outputs = torch.cat([lstm_outputs, gat_outputs], dim=-1)
        outputs = outputs.squeeze(0)
        # OUTPUT
        outputs = self.Dropout(outputs)
        outputs = self.output_linear(outputs)
        return outputs  # [node_num , label_num] logit


if __name__ == "__main__":
    from codes2.utils.dataset import BiLSTMDataset
    fileholder = "../../data2/"
    dataLoader = BiLSTMDataset(fileholder, 0.3, 1, "train", vocab_filename="../vocab.txt", shuffle=False, use_cuda=False, device=None)

    model = BiLSTM_Graph(
        char_emb_dim=50,
        lstm_hidden_dim=128,
        lstm_layer=1,
        char_alphabet_size=5563,
        pretrained_char_emb_filename=None,
        dropout_rate=0.1,
        label_size=3,
        gat_hidden_dim=128,
        alpha=0.1,
        gat_nheads=4,
        gat_layer=2
    )
    for data in dataLoader:
        (batch_char, batch_len, mask), adj, label = data
        outputs = model(batch_char, batch_len, mask, adj)
        print(outputs.size())
