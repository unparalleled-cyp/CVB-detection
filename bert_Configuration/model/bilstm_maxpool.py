import torch.nn as nn
import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F


class BLSTM_MAXPOOL(nn.Module):
    def __init__(self,
                 char_emb_dim,
                 hidden_dim,
                 lstm_layer,
                 char_alphabet_size,
                 pretrained_char_emb_filename,
                 dropout_rate,
                 label_size):
        """
        初始化
        :param char_emb_dim: 字向量的维度
        :param hidden_dim: LSTM隐藏层的维度
        :param lstm_layer: LSTM层数
        :param char_alphabet_size: 字典大小
        :param pretrained_char_emb_filename: 保存预训练字向量的文件，为空就说明不需要导入预训练的字向量
        :param dropout_rate: dropout率
        :param label_size: 标签数目
        :param use_cuda: 是否使用gpu
        :param device: 使用哪个设备
        """
        super(BLSTM_MAXPOOL, self).__init__()
        print("build BLSTM_MAXPOOL model...")
        self.name = "BLSTM_MAXPOOL"
        self.char_embeddings = nn.Embedding(char_alphabet_size, char_emb_dim)
        self.char_dropout = nn.Dropout(dropout_rate)
        if pretrained_char_emb_filename is not None:
            # 先读取文件
            pretrained_char_emb = self.read_pretrained_char_emb(pretrained_char_emb_filename)
            self.char_embeddings.weight.data.copy_(torch.from_numpy(pretrained_char_emb))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(char_alphabet_size, char_emb_dim)))
        lstm_hidden = hidden_dim // 2
        self.lstm = nn.LSTM(char_emb_dim, lstm_hidden, num_layers=lstm_layer, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(dropout_rate)
        self.final_dense = nn.Linear(hidden_dim, label_size)
        self.final_dropout = nn.Dropout(dropout_rate)
        self.reset_parameters()
        self.to_double()

        # self.device = device
        # if use_cuda:
        #     self.to_cuda()

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

    def to_double(self):
        self.char_embeddings = self.char_embeddings.to(torch.double)
        self.char_dropout = self.char_dropout.to(torch.double)
        self.lstm = self.lstm.to(torch.double)
        self.drop_lstm = self.drop_lstm.to(torch.double)
        self.final_dense = self.final_dense.to(torch.double)
        self.final_dropout = self.final_dropout.to(torch.double)

    def to_cuda(self):
        self.char_embeddings = self.char_embeddings.to(self.device)
        self.char_dropout = self.char_dropout.to(self.device)
        self.lstm = self.lstm.to(self.device)
        self.drop_lstm = self.drop_lstm.to(self.device)
        self.final_dense = self.final_dense.to(self.device)
        self.final_dropout = self.final_dropout.to(self.device)

    def reset_parameters(self):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0_reverse)
        nn.init.orthogonal_(self.lstm.weight_ih_l0_reverse)
        nn.init.orthogonal_(self.final_dense.weight)
        nn.init.constant_(self.final_dense.bias, 0)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_char, batch_len):
        # print(self.char_embeddings, batch_char.device)
        # print(batch_char.device, self.char_embeddings.weight.device)
        embeds = self.char_embeddings(batch_char)
        embeds = self.char_dropout(embeds)
        # embeds_pack = pack_padded_sequence(embeds, batch_len, batch_first=True)  # 本身的embeds是已经pad了，如果将这个导入到lstm中，那么那些无用的Pad会造成误差，所以要pack
        # out_packed, (_, _) = self.lstm(embeds_pack)
        # lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True)  # pack后的embeds经过lstm之后要进行pad
        lstm_feature = self.lstm(embeds)[0]
        lstm_feature = self.drop_lstm(lstm_feature)
        return lstm_feature

    def _mask(self, input_tensor, mask):
        """
        进行mask操作
        """
        input_tensor = input_tensor - (1 - mask) * 1e8  # 将本身为1的不变，为0的减少至负无穷
        return input_tensor

    def _get_feature(self, batch_char, batch_len):
        lstm_feature = self._get_lstm_features(batch_char, batch_len)  # 字级的embedding+lstm提取特征
        return lstm_feature

    def _get_final_features(self, lstm_feature, mask):
        features = self._mask(lstm_feature, mask)
        features = features.transpose(-1, -2).contiguous()
        final_features = F.max_pool1d(features, [features.size(-1)]).squeeze(-1)
        return final_features
        # final_features = self.final_dropout(final_features)
        # final = self.final_dense(final_features)
        # return final

    def forward(self, batch_char, batch_len, mask):
        """

        :param batch_char:  [node_num, seq_len,]
        :param batch_len: [node_num, ]
        :param mask: [node_num, seq_len]
        :return:
        """
        # batch_len = batch_len.cpu()
        mask = torch.unsqueeze(mask, dim=-1)
        lstm_feature = self._get_feature(batch_char, batch_len) # kjtodo
        feature = self._get_final_features(lstm_feature, mask)
        return feature


if __name__ == "__main__":
    batch_char = torch.tensor([[17,  17,  8,  5,  0, 0,  0,  0, 0,  0, 0,  0],
                                [ 3,  2,  2, 18,  5, 10, 14,  2,  1,  9, 14,  2],
                                [12, 18, 16, 10,  9,  8, 11,  2,  1,  0,  0, 0]])
    batch_len = torch.tensor([4, 12, 9])
    mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    ids = np.argsort(batch_len.numpy())
    ids = ids[-1::-1].copy()
    batch_char, batch_len, mask = batch_char[ids], batch_len[ids], mask[ids]
    module = BLSTM_MAXPOOL(
        char_emb_dim = 10,
        hidden_dim = 30,
        lstm_layer = 2,
        char_alphabet_size = 100,
        pretrained_char_emb_filename = None,
        dropout_rate = 0.2,
        label_size = 4
    )
    # 长度必须是降序
    res = module(batch_char, batch_len, mask)
    print(res)
