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
from tensorboardX import SummaryWriter


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def compute_metrics(labels, predicts,  args):
    """
    calculate accuracy, F1, confusion-matrix
    precision[i] = number of correct predictions for class i / number of predictions for class i
    recall[i] = number of correct predictions for class i / number of true labels for class i
    1 / f1 = 1 / precision + 1 / recall -> f1 = 2*precision*recall/(precision+recall)
    confusion-matrix[i][j] represents the number of instances of class i predicted as class j
    :param predicts: predicted labels
    :param labels: true labels
    :return:
    """
    # calculate accuracy
    acc = sum(predicts == labels) / len(predicts)

    # calculate precision
    precisions = []
    for i in range(args.label_size):
        right = 0
        total = 0
        for sample_l, sample_p in zip(labels, predicts):
            if (sample_l == i) and (sample_p == i):
                right += 1
            if sample_p == i:
                total += 1
        precisions.append(right / max(total, 0.001))

    # calculate recall
    recalls = []
    for i in range(args.label_size):
        right = 0
        total = 0
        for sample_l, sample_p in zip(labels, predicts):
            if (sample_l == i) and (sample_p == i):
                right += 1
            if sample_l == i:
                total += 1
        recalls.append(right / max(total, 0.001))

    # calculate F1
    f1s = [2*p*r/(max(p+r, 0.001)) for p, r in zip(precisions, recalls)]
    F1 = sum(f1s) / len(f1s)

    # calculate confusion matrix
    cm = confusion_matrix(labels, predicts)

    return acc, F1, cm, f1s


def evaluate(model, dataloader, loss_func, args):
    """
    Evaluation result: loss value, accuracy (%), F1 score (%), confusion matrix
    :param model:
    :param dataloader:
    :param args:
    :return:
    """
    model.eval()
    predicts = []  
    total_labels = []

    total_loss = 0.

    tmp_features, tmp_labels = [], []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            cid, (batch_char, batch_len, mask), adj, labels, user_infos = data
            torch.cuda.empty_cache()
            feature = model(batch_char, batch_len, mask, adj, user_infos)
            ids = labels != -1  
            labels = labels[ids]
            feature = feature[ids]
            tmp_features.append(feature)
            tmp_labels.append(labels)
            if len(tmp_features) >= args.batch_size:
                tmp_features, tmp_labels = torch.cat(tmp_features), torch.cat(tmp_labels)
                loss = loss_func(tmp_labels, tmp_features)

                # update loss and accuracy
                total_loss += (loss.item() * len(tmp_features))
                predic = torch.max(tmp_features, 1)[1].cpu().numpy()
                predicts.extend(predic)
                total_labels.extend(tmp_labels.cpu().numpy())
                
                tmp_features, tmp_labels = [], []

        if tmp_features:
            tmp_features, tmp_labels = torch.cat(tmp_features), torch.cat(tmp_labels)
            loss = loss_func(tmp_labels, tmp_features)

            # update loss and accuracy
            total_loss += (loss.item() * len(tmp_features))
            predic = torch.max(tmp_features, 1)[1].cpu().numpy()
            predicts.extend(predic)
            total_labels.extend(tmp_labels.cpu().numpy())

    labels = np.array(total_labels)
    avg_loss = total_loss / len(predicts)

    # calculate acc, f1, confusion-matrix
    acc, F1, cm, f1s = compute_metrics(labels=labels,
                                  predicts=predicts,
                                  args=args)

    return avg_loss, acc*100, F1*100, cm, f1s


def train(model, train_dataloader, val_dataloader, loss_func, args):
    """
    Operation: the following parts are added
        1. During training, update parameters only when the number of labels in the data reaches a certain threshold
    :param model: Graph model
    :param train_dataloader: training data generator
    :param val_dataloader: validation data generator
    :param loss_func: loss function
    :param args: arguments
    :return:
    """
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.l2_penalty)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.l2_penalty)

    best_val_f1 = -1
    for epoch_id in range(args.max_epoch):
        optimizer = lr_decay(optimizer, epoch_id, args.lr_decay, args.lr)
        model.train()
        model.zero_grad()

        total_loss = 0.
        right_count = 0
        total_count = 0
        start_time = time.time()

        tmp_features, tmp_labels = [], []
        #i = 0
        #for data in train_dataloader:
        batch_count_nodes = 0
        batch_count = 0
        for i, data in tqdm(enumerate(train_dataloader)):
            # if i == 1511: print(111)
            torch.cuda.empty_cache()
            cid, (batch_char, batch_len, mask), adj, labels, user_infos = data
            print('graph:', i, len(labels[labels != -1]), '/', len(labels))
            #feature = None
            #torch.cuda.empty_cache()
            #print('StartTime:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), i, len(tmp_features))
            #if len(tmp_features) >= args.batch_size:
            if len(labels) > args.batch_size:  
                if True:
                    print('超大图', len(labels[labels != -1]),'/',len(labels),'已跳过')
                else: 
                    batch_count += 1
                    feature = model(batch_char, batch_len, mask, adj, user_infos)
                    ids = labels != -1  
                    tmp_labels = labels[ids]
                    tmp_features = feature[ids]
                    #tmp_features.append(feature)
                    #tmp_labels.append(labels)
                    #tmp_features, tmp_labels = torch.cat(tmp_features), torch.cat(tmp_labels)
                    loss = loss_func(tmp_labels, tmp_features)
                    torch.cuda.empty_cache()
                    loss.backward()
                    optimizer.step()
                    model.zero_grad()
                    # update loss and accuracy
                    total_loss += (loss.item() * len(tmp_features))
                    predic = torch.max(tmp_features, 1)[1].cpu().numpy()
                    tmp_right = np.sum(tmp_labels.cpu().numpy() == predic)
                    right_count += tmp_right
                    total_count += len(tmp_labels)

                    avg_loss = total_loss / total_count
                    avg_acc = right_count / total_count * 100

                    if batch_count % args.print_iter == 0 or batch_count == 1:
                        print("Epoch: [%i / %i], iteration: %i; Time: %.2fs; loss: %.4f; acc: %.2f%%" % (
                            epoch_id+1, args.max_epoch, i+1, time.time()-start_time, avg_loss, avg_acc))
                        writer.add_scalar("acc/training", avg_acc, epoch_id*1000+batch_count)
                        writer.add_scalar("loss/training", avg_loss, epoch_id*1000+batch_count)
                        sys.stdout.flush()

                    tmp_features, tmp_labels = [], []
            elif batch_count_nodes + len(labels) > args.batch_size: 
                batch_count += 1
                torch.cuda.empty_cache()
                tmp_features, tmp_labels = torch.cat(tmp_features), torch.cat(tmp_labels)
                loss = loss_func(tmp_labels, tmp_features)
                loss.backward()  
                optimizer.step()
                model.zero_grad()

                # update loss and accuracy
                total_loss += (loss.item() * len(tmp_features))
                predic = torch.max(tmp_features, 1)[1].cpu().numpy()
                tmp_right = np.sum(tmp_labels.cpu().numpy() == predic)
                right_count += tmp_right
                total_count += len(tmp_labels)

                avg_loss = total_loss / total_count
                avg_acc = right_count / total_count * 100

                if batch_count % args.print_iter == 0 or batch_count == 1:
                    print("Epoch: [%i / %i], iteration: %i; Time: %.2fs; loss: %.4f; acc: %.2f%%" % (
                        epoch_id+1, args.max_epoch, i+1, time.time()-start_time, avg_loss, avg_acc))
                    writer.add_scalar("acc/training", avg_acc, epoch_id*1000+batch_count)
                    writer.add_scalar("loss/training", avg_loss, epoch_id*1000+batch_count)
                    sys.stdout.flush()

                tmp_features, tmp_labels = [], []
                batch_count_nodes = len(labels)
                feature = model(batch_char, batch_len, mask, adj, user_infos)
                ids = labels != -1  
                labels = labels[ids]
                feature = feature[ids]
                tmp_features.append(feature)
                tmp_labels.append(labels)
            else:  
                batch_count_nodes += len(labels)
                feature = model(batch_char, batch_len, mask, adj, user_infos)
                ids = labels != -1  
                labels = labels[ids]
                feature = feature[ids]
                tmp_features.append(feature)
                tmp_labels.append(labels)

        if len(tmp_features) > 0: 
            batch_count += 1
            tmp_features, tmp_labels = torch.cat(tmp_features), torch.cat(tmp_labels)
            loss = loss_func(tmp_labels, tmp_features)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # update loss and accuracy
            total_loss += (loss.item() * len(tmp_features))
            predic = torch.max(tmp_features, 1)[1].cpu().numpy()
            tmp_right = np.sum(tmp_labels.cpu().numpy() == predic)
            right_count += tmp_right
            total_count += len(tmp_labels)

            avg_loss = total_loss / total_count
            avg_acc = right_count / total_count * 100

            if batch_count % args.print_iter == 0 or batch_count == 1:
                print("Epoch: [%i / %i], iteration: %i; Time: %.2fs; loss: %.4f; acc: %.2f%%" % (
                    epoch_id+1, args.max_epoch, i+1, time.time()-start_time, avg_loss, avg_acc))
                writer.add_scalar("acc/training", avg_acc, epoch_id*1000+batch_count)
                writer.add_scalar("loss/training", avg_loss, epoch_id*1000+batch_count)
                sys.stdout.flush()
        
        val_loss, val_acc, val_f1, val_cm, val_f1s = evaluate(model, val_dataloader, loss_func, args)
        writer.add_scalar("acc/val", val_acc, epoch_id+1)
        writer.add_scalar("f1/val", val_f1, epoch_id+1)
        writer.add_scalar("loss/val", val_loss, epoch_id+1)

        if val_f1 > best_val_f1:
            print("current val f1: [%.2f%%], exceed previous best val f1: [%.2f%%]" % (val_f1, best_val_f1))
            if not os.path.exists(args.param_stored_fileholder):
                os.makedirs(args.param_stored_fileholder)
            model_name = os.path.join(args.param_stored_fileholder, "best.model")

            torch.save(model.state_dict(), model_name)
            best_val_f1 = val_f1
        print("current val acc: %.2f%%, current val f1: %.2f%%, current val loss: %.4f, best val f1: %.2f%%" %
              (val_acc, val_f1, val_loss, best_val_f1))
        print("val confusion matrix: ")
        print(val_cm)
        print("val f1s: ", val_f1s)

        if epoch_id % args.evaluate_train == 0:
            train_loss, train_acc, train_f1, train_cm, train_f1s = evaluate(model, train_dataloader, loss_func, args)
            writer.add_scalar("acc/train", train_acc, epoch_id + 1)
            writer.add_scalar("f1/train", train_f1, epoch_id + 1)
            writer.add_scalar("loss/train", train_loss, epoch_id + 1)
            print("current train acc: %.2f%%, current train f1: %.2f%%, current train loss: %.4f" %
                  (train_acc, train_f1, train_loss))
            print("train confusion matrix: ")
            print(train_cm)
            print("train f1s: ", train_f1s)

        gc.collect()





if __name__ == '__main__':
    startTime = time.localtime()
    print('StartTime:', time.strftime("%Y-%m-%d %H:%M:%S", startTime))
    
    args, unparsed = get_args()

    for arg in vars(args):
        print(arg, ":",  getattr(args, arg))

    writer = SummaryWriter(log_dir=args.log_path + '/' + time.strftime('%Y-%m-%d_%H%M%S', startTime))


    # if args.use_gpu:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for _ in range(1):
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
    
        train_dataloader = BiLSTMDataset(fileholder=args.fileholder,
                                        #  dev_rate=args.dev_rate,
                                         seed=args.random_seed,
                                         mode="train",
                                         vocab_filename=args.vocab_filename,
                                         shuffle=args.shuffle,
                                         use_cuda=args.use_gpu,
                                         device=device)

        val_dataloader  =  BiLSTMDataset(fileholder=args.fileholder,
                                        #  dev_rate=args.dev_rate,
                                         seed=args.random_seed,
                                         mode="val",
                                         vocab_filename=args.vocab_filename,
                                         shuffle=args.shuffle,
                                         use_cuda=args.use_gpu,
                                         device=device)
        print('LoadData:', time.mktime(time.localtime()) - time.mktime(startTime))
        
        print('train_dataloader')
        t = []
        for i in train_dataloader.labels:
            t.extend(i[1:])
        print(pd.Series(t).value_counts())

        print('val_dataloader')
        t = []
        for i in val_dataloader.labels:
            t.extend(i[1:])
        print(pd.Series(t).value_counts())
        if args.mode == "train":
            train(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, loss_func=loss_func, args=args)
    
        model_file = os.path.join(args.param_stored_fileholder, "best.model")
        os.system('copy {} {}'.format(model_file, os.path.join(args.param_stored_fileholder, time.strftime("%Y-%m-%d_%H%M%S.model", startTime))))
        print("Model file: ", model_file)
        if not os.path.exists(model_file):
            raise Exception(f"不存在模型文件: {model_file}")
        model.load_state_dict((torch.load(model_file, map_location=lambda storage, loc: storage)))
    
        print("Final result")
    
        train_loss, train_acc, train_f1, train_cm, train_f1s = evaluate(model, train_dataloader, loss_func, args)
        print("current train acc: %.2f%%, current train f1: %.2f%%, current train loss: %.4f" %
              (train_acc, train_f1, train_loss))
        print("train confusion matrix: ")
        print(train_cm)
        print("train f1s: ", train_f1s)
    
        val_loss, val_acc, val_f1, val_cm, val_f1s = evaluate(model, val_dataloader, loss_func, args)
        print("current val acc: %.2f%%, current val f1: %.2f%%, current val loss: %.4f" %
              (val_acc, val_f1, val_loss))
        print("val confusion matrix: ")
        print(val_cm)
        print("val f1s: ", val_f1s)
    
    endTime = time.localtime()
    print('EndTime:', time.strftime("%Y-%m-%d %H:%M:%S", endTime))
    timeDiff = time.mktime(endTime)-time.mktime(startTime)
    print("SpendTime: {}h {}min {}s ".format(int(timeDiff // 3600), int(timeDiff % 3600 // 60), int(timeDiff % 3600 % 60)))
