# -*- coding: utf-8 -*-
"""
@File: BiLSTM_Graph.py 
@Time : 2024/11/28 17:27

@dec: Bert-Gat
"""

import torch
from torch import nn

from transformers import BertModel
from model.gatlayer import GAT
from model.user_info_embedding import UserInfoEmbedding

from utils.dataset_bert import BertDataset

class Bert_Gat(nn.Module):
    """
    bert + gat model
    1. Use BERT to encode the input sequences to obtain feature vectors.
    2. Use UserInfoEmbedding to process user-specific features.
    3. Combine BERT outputs with user embeddings, and pass them to GAT for graph processing.
    """
    def __init__(self, char_alphabet_size, pretrained_char_emb_filename,
                 gat_hidden_dim, bert_model_name):
        super(Bert_Gat, self).__init__()

        # Load Pretrained BERT Model
        self.bert = BertModel.from_pretrained(bert_model_name)
#         self.bert_hidden_dim = bert_hidden_dim  # BERT hidden size

        # Parameters
        rate = 1/8
        self.gat_hidden_dim = gat_hidden_dim
        self.pretrained_char_emb_filename = pretrained_char_emb_filename


        # UserInfoEmbedding
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
             char_dim = 50,
             hidden_dim = 32,
             lstm_layer = 2,
             dropout_rate = 0.3,
             pretrained_char_emb_filename = pretrained_char_emb_filename)

        # Graph Attention Network (GAT)
        self.GAT = GAT(
            nfeat= 800, #int(self.bert_hidden_size + 32),
            nhid=gat_hidden_dim,
            nclass=3,
            dropout=0.3,
            alpha=0.1,
            nheads=4,
            layer=2)
        
        # Dropout and Output Layer
        self.Dropout = nn.Dropout(0.3)
        self.output_linear = nn.Linear(1056, 3).to(torch.double) #int(gat_hidden_dim*(2+rate))
        

    def forward(self, input_ids, attention_mask, adj, user_infos): # , token_type_ids
            
            # BERT Ecoding
            bert_outputs = self.bert(
                 input_ids=input_ids, attention_mask=attention_mask) #, token_type_ids=token_type_ids
            bert_seq_output = bert_outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
            bert_pooled_output = bert_outputs.pooler_output  # Optional: [batch_size, hidden_dim]

            # UserInfoEmbedding
            user_info_outputs = self.UserInfo(user_infos)

            lstm_outputs = torch.cat([bert_pooled_output, user_info_outputs], dim=-1)
            lstm_outputs = lstm_outputs.unsqueeze(0)
            # Combine BERT and UserInfo Outputs
            # combined_features = torch.cat([bert_seq_output, user_info_outputs.unsqueeze(1).repeat(1, bert_seq_output.size(1), 1)], dim=-1)  # [batch_size, seq_len, combined_dim]
            ## combined_features = torch.cat([bert_seq_output, user_info_outputs.unsqueeze(1)], dim=-1)
            ## combined_features = combined_features.unsqueeze(0)  # [1, node_num, dim]

            # GAT 编码
            adj = adj.unsqueeze(0)  # [1, node_num, node_num]
            gat_outputs = self.GAT(lstm_outputs, adj)
            ## gat_outputs = self.GAT(combined_features, adj)

            outputs = torch.cat([lstm_outputs, gat_outputs], dim=-1)
            outputs = outputs.squeeze(0)
            # Final output
            # outputs = torch.cat([combined_features, gat_outputs], dim=-1)
            # outputs = outputs.squeeze(0)
            outputs = self.Dropout(outputs)
            outputs = self.output_linear(outputs)

            return outputs

if __name__ == "__main__":
    from transformers import BertTokenizer
    from dataset_bert import BertDataset
    fileholder = "../../dataset/data_bak/1.5-s125/"
    dataLoader = BertDataset(fileholder, seed=0, mode="train", vocab_filename="../../embeddings/vocab.txt", shuffle=False, use_cuda=False, device=None, bert_model="../../bert-base-chinese")

    model = Bert_Gat(
#         char_emb_dim=50,
        char_alphabet_size=5789,
        pretrained_char_emb_filename="../../embeddings/weights.txt",
        gat_hidden_dim=256,
        bert_model_name="../../bert-base-chinese",
    )

        
    for data in dataLoader:
        (input_ids, attention_mask), adj, labels, user_infos = data #, token_type_ids

        outputs = model(input_ids, attention_mask, adj, user_infos) # , token_type_ids
#         print(f"Model outputs shape: {outputs.size()}")
        
#         print(f"Adj shape: {adj.shape}")  # 打印 adj 的形状
#         print(f"input_ids shape: {input_ids.shape}")
#         print(f"attention_mask shape: {attention_mask.shape}")
#         print(f"token_type_ids shape: {token_type_ids.shape}")
#         print(f"adj shape: {adj.shape}")
#         print(f"user_infos shape: {user_infos.shape}")