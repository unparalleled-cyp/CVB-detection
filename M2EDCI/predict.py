# -*- coding: utf-8 -*-
"""
@File: predict.py 
@Time : 2023/6/7 17:13

@dec:
"""


from utils.dataset import BiLSTMDataset
from utils.config import get_args
from model.BiLSTM_Graph import BiLSTM_Graph as module

import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.nn import functional as F
import time
import sys
import gc
from tqdm import tqdm
from torch import nn
from torch.nn.functional import one_hot
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle
import json

def predict_2(model, dataloader, loss_func, args):
    """
    Evaluation result: loss value, accuracy (%), F1 score (%), confusion matrix
    :param model:
    :param dataloader:
    :param args:
    :return:
    """
    model.eval()
    cids = []
    prob1 = []
    prob2 = []
    prob3 = []
    predict = []
    label = []

    # ccc = 0
    # B_posts = []
    # B_cids = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            torch.cuda.empty_cache()
            cid, (batch_char, batch_len, mask), adj, labels, user_info = data
            cid = [str(c) for c in cid]
            # if len(cid) > 200:
            #     ccc += 1
            #     print(ccc, len(cid))
            #     B_posts.append(cid[0])
            #     B_cids.extend(cid[1:])
            # continue
            cids.extend(cid)
            feature = model(batch_char, batch_len, mask, adj, user_info)
            prob = torch.softmax(feature, dim=-1)
            prob1.extend([p[0].item() for p in prob])
            prob2.extend([p[1].item() for p in prob])
            prob3.extend([p[2].item() for p in prob])
            predict.extend(torch.max(prob, 1)[1].cpu().numpy())
            label.extend(labels.cpu().numpy())

    # print(len(B_posts), len(B_cids))
    # B_posts = list(set(B_posts))
    # B_cids = list(set(B_cids))
    # print(len(B_posts), len(B_cids))
    # with open('BigGraph_Res2.json', 'w') as f:
    #     json.dump({
    #         "B_posts": B_posts,
    #         "B_cids": B_cids
    #     }, f)

    return {
        "cids" : cids,
        "prob1": prob1,
        "prob2": prob2,
        "prob3": prob3,
        "predict": predict,
        "label": label
    }


if __name__ == '__main__':
    MODEL_FILE = '2022-12-24_214043'

    startTime = time.localtime()
    print('StartTime:', time.strftime("%Y-%m-%d %H:%M:%S", startTime))

    args, unparsed = get_args()
    args.mode = 'predict'
    args.shuffle = False
    args.batch_size = 200
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))

    # if args.use_gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    if args.use_gpu:
        device_ids = list(map(int, args.visible_gpu.split(",")))
        device = torch.device("cuda:%i" % device_ids[0])
    else:
        device_ids = []
        device = torch.device("cpu")
    print(device)

    loss_func = lambda label, feature: F.cross_entropy(feature, label)
    # sparse_categorical_cross_entropy(
    #     gamma=gamma, categorical_num=args.label_size, alpha=alpha, epsilon = 1e-6, device=device, args=args)
    # if args.use_gpu:
    #     loss_func = loss_func.to(device)
    # seed = args.random_seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    model = module(
        char_emb_dim=args.char_emb_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layer=args.lstm_layer,
        char_alphabet_size=args.char_alphabet_size,
        pretrained_char_emb_filename=args.pretrained_char_emb_filename,
        dropout_rate=args.dropout_rate,
        gat_dropout_rate=args.gat_dropout_rate,
        label_size=args.label_size,
        gat_hidden_dim=args.gat_hidden_dim,
        alpha=args.alpha,
        gat_nheads=args.gat_nheads,
        gat_layer=args.gat_layer
    )

    print(model)
    if len(device_ids) >= 2:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        model = model.to(device)

    pred_dataloader = BiLSTMDataset(fileholder=args.fileholder,
                                #    dev_rate=1.0,
                                   seed=args.random_seed,
                                   mode="predict",
                                   vocab_filename=args.vocab_filename,
                                   shuffle=args.shuffle,
                                   use_cuda=args.use_gpu,
                                   device=device)
    t = []
    for i in pred_dataloader.labels:
        t.extend(i[1:])
    print(pd.Series(t).value_counts())
    model_file = os.path.join(args.param_stored_fileholder, "{}.model".format(MODEL_FILE))
    print("Model file: ", model_file)
    if not os.path.exists(model_file):
        raise Exception(f"不存在模型文件: {model_file}")
    model.load_state_dict((torch.load(model_file, map_location=lambda storage, loc: storage)))

    print("Final result")

    results = predict_2(model, pred_dataloader, loss_func, args)
    results = pd.DataFrame(results)
    with open('./{}/{}.pkl'.format(args.param_stored_fileholder, MODEL_FILE), 'wb') as fid:
        pickle.dump(results, fid)
    with pd.ExcelWriter("./{}/{}.xlsx".format(args.param_stored_fileholder, MODEL_FILE)) as f:
        results.to_excel(excel_writer = f)

    endTime = time.localtime()
    print('EndTime:', time.strftime("%Y-%m-%d %H:%M:%S", endTime))
    timeDiff = time.mktime(endTime)-time.mktime(startTime)
    print("SpendTime: {}h {}min {}s ".format(int(timeDiff // 3600), int(timeDiff % 3600 // 60), int(timeDiff % 3600 % 60)))

