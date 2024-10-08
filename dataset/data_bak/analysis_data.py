# -*- coding: utf-8 -*-
"""
@File: analysis_data.py 
@Time : 2023/5/4 18:41

@dec: 
"""


import pandas as pd
import glob
import os
import json
import re

import win32file
from tqdm import tqdm
from queue import Queue
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, cpu_count
import numpy as np
from pprint import pprint
from clean_data import clean_string, cut_sentence


# filename = r"data2/26-4602660827432508-07705-00967.xlsx"

deal_num = 0

def read_excel(filename, sheet_name=0):
    '''读取excel'''
    df = pd.read_excel(filename, sheet_name=sheet_name)
    return df


def read_dir(fileholder):
    '''读取data目录下的excel列表'''
    fileholder = os.path.join(fileholder, "*.xlsx")
    filenames = glob.glob(fileholder)
    filenames = [f for f in filenames if "dataAll.xlsx" not in f]
    return filenames


def recursive(links):
    """
    遍历每个节点，获得每个图有哪些节点
    :param links: links[k]表示k的邻居
    :return:
    1、遍历links中的每个节点记为node，
    同时要创建一个traversed的集合，如果新的节点在traversed中，则说明已经遍历过，无需再遍历，否则需要将图的个数+1；
    2、创建一个队列q，将node加入进来；
    3、遍历q中的每个元素n，如果n在traversed中，就pass；否则遍历links[n]，如果没在traversed中，就加入到q中；
    """
    traversed = set()
    graph_num = 0
    graphs = []

    for node in links:
        if node not in traversed:  # 如果node已经遍历过，就不算
            tmp_nodes = []  # 保存这个图中有哪些节点
            graph_num += 1
            q = Queue()
            q.put(node)  # 初始节点
            while q.qsize():  # 如果q没有元素，就会跳出while循环
                q_size = q.qsize()
                # 遍历q中的每个节点
                for _ in range(q_size):
                    n = q.get()
                    if n not in traversed:
                        tmp_nodes.append(n)
                        traversed.add(n)
                        # 先找到n的邻接节点
                        for link_n in links[n]:
                            q.put(link_n)
            tmp_nodes = list(set(tmp_nodes))
            tmp_nodes.sort()
            graphs.append(tmp_nodes)

    return graphs


def statistic_graph(df):
    cids = df["cid"]
    fNodes = df["fNode"]
    assert len(cids) == len(fNodes)

    # 去掉根节点，不考虑所有跟根节点相连的节点
    root = cids[0]
    links = dict()  # links[k]为k邻居
    for i in range(1, len(cids)):
        cid, fNode = cids[i], fNodes[i]
        if cid not in links:
            links[cid] = set()
        if fNode not in links:
            links[fNode] = set()
        # 所有的节点都跟自己连接
        links[cid].add(cid)
        links[fNode].add(fNode)
        if fNode != root:
            links[cid].add(fNode)
            links[fNode].add(cid)

    # 获得各个图，获得各个图的节点
    graphs = recursive(links)
    # 将根节点插入进来
    new_graphs = []
    for graph in graphs:
        new_graphs.append(graph)
    return new_graphs


def analysis_data_one_file(filename):
    """
    先读取文件，再数一下各个图有多少个节点
    :param filename: 文件名
    :return:
    """
    global deal_num
    print("Processing %s, in %i-th" % (filename, deal_num))
    deal_num += 1
    df = read_excel(filename)
    graphs = statistic_graph(df)
    tmp = "\t".join([filename, ] + list(map(str, [len(graph) for graph in graphs])))
    print("Finished %s " % filename)
    return tmp


def analysis_data(fileholder, graph_file):
    """
    分析.xlsx后缀的文件，使用多进程来做吧
    先读取文件，再数一下各个图有多少个节点
    :param fileholder: 文件夹
    graph_file: graph保存的文件
    :return:
    """
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    processes = []
    # analysis_data_one_file(filenames[0])
    with ProcessPoolExecutor(cpu_count() if cpu_count() <=61 else 61) as pool:
        for filename in tqdm(filenames):
            processes.append(pool.submit(analysis_data_one_file, filename))

    results = []
    for process in processes:
        results.append(process.result())

    # 保存
    with open(graph_file, "w", encoding="utf-8") as f:
        for s in results:
            f.write(s + "\n")


def analysis_graph_file(graph_file, graph_file_res):
    """
    读取graph_file数据，进行分析
    :param graph_file: graph文件
    :return:
    """
    nums = []
    with open(graph_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            nums.extend(list(map(int, line[1:])))
    nums.sort(reverse=True)
    print(nums)
    # 保存
    with open(graph_file_res, "w", encoding="utf-8") as f:
        for s in nums:
            f.write(str(s) + "\n")


def statistic_com_nums(file, garph_size_limit):
    count = 0
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = int(line.strip())
            if line < garph_size_limit: break
            count += line
    print('构成图的评论一共有', count, '条')
    return count


def statistic_lengths(fileholder):
    """
    统计文本长度
    :param fileholder:文件夹名
    :return:
    """
    lengths = []
    lengths_post = []
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    count_nan = 0
    labels = {}

    for name in tqdm(filenames):
        df = read_excel(name)
        l = int(df["label"][0])
        if l not in labels:
            labels[l] = 0
        labels[l] += 1
        lent = [len(cut_sentence(clean_string(c))) if str(c) != "nan" else 0 for c in df["content"][1:]] # 去除post的所有comment，content长度
        for l in lent:
            if l == 0:
                count_nan += 1
        lengths.extend(lent)
        lengths_post.append(len(cut_sentence(clean_string(df["content"][0]))))
    lengths.sort(reverse=True)
    lengths_post.sort(reverse=True)

    print('post label统计：', labels)
    print('com最长/post最长/com平均长度/post平均长度', lengths[0], lengths_post[0], sum(lengths) / len(lengths), sum(lengths_post) / len(lengths_post))
    print('content为NaN：', count_nan)
    # 保存
    with open("data_ana.txt", "a", encoding="utf-8") as f:
        f.write('post label统计：' + str(labels) + "\n")
        f.write('com最长：' + str(lengths[0]) + "\n")
        f.write('post最长：' + str(lengths_post[0]) + "\n")
        f.write('com平均长度：' + str(sum(lengths) / len(lengths)) + "\n")
        f.write('post平均长度：' + str(sum(lengths_post) / len(lengths_post)) + "\n")
        f.write('content为NaN：' + str(count_nan) + "\n\n")


def statistic_label(fileholder):
    """
    1、统计各个文件夹中标签数据占的比重
    2、统计不同标签的数量比重
    :param fileholder:
    :return:
    """
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    rates_0 = []  # 统计各个文件中标签为0所占的比重
    counts = [0, 0, 0, 0, 0]  # 统计所有文件加和之后各标签所占的比重 label为0 1 2 3 4的数量
    root_count = 0
    for name in tqdm(filenames):
        df = read_excel(name)
        tmp_counts = [sum(df["label"][1:] == i) for i in range(5)]
        root_count += (df["label"][0] != 0)
        rates_0.append(tmp_counts[0] / sum(tmp_counts))
        for i in range(5):
            counts[i] += tmp_counts[i]
    rates_0.sort(reverse=True)
    print('所有评论总的label分布0-4：', counts)
    print('每个post label为0比率排序：', rates_0)
    # 保存
    with open("data_ana.txt", "a", encoding="utf-8") as f:
        f.write('label分布0-4：' + str(counts) + "\n")
        f.write('label为0比率排序：\n')
        for item in rates_0:
            f.write(str(item) + "\n")
        f.write("\n")


def get_vocab(fileholder, vocab_filename):
    """
    统计filehoder中各文件的文本中字的分布情况
    :param fileholder:
    :param vocab_filename: 字典保存的地址
    :return:
    """
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    vocabs = dict()
    for name in tqdm(filenames):
        df = read_excel(name)
        for string in df["content"]:
            if str(string) != "nan":
                string = cut_sentence(clean_string(string))
                for c in string:
                    c = c.strip()
                    if c:
                        if c not in vocabs:
                            vocabs[c] = 0
                        vocabs[c] += 1
    sort_items = list(vocabs.items())
    sort_items.sort(key=lambda x: x[1], reverse=True)
    vocabs = {v: count for v, count in sort_items}
    # 保存字典
    with open(vocab_filename, "w", encoding="utf-8") as f:
        for v, n in sort_items:
            f.write(v + "\n")


def read_vocab_filename(vocab_filename):
    """
    从vocab字典中读取字，变成字典格式
    :param vocab_filename:
    :return:
    """
    vocabs = {"PAD": 0, "UNK": 1}
    with open(vocab_filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line in vocabs:
                raise Exception("字[%i]已经在字典中" % line)
            vocabs[line] = len(vocabs)
    return vocabs


def read_weight_filename(weight_filename):
    """
    读取权重
    :param weight_filename:
    :return:
    """
    weights = dict()
    with open(weight_filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            if line:
                c = line[0]
                nums = list(map(float, line[1:]))
                assert len(nums) == 50
                if c and c not in weights:
                    weights[c] = nums
    return weights


def random_weight(embedding_dim):
    """
    随机生辰权重
    :return:
    """
    scale = np.sqrt(3.0 / embedding_dim)
    res = np.random.uniform(-scale, scale, [embedding_dim])
    return res.tolist()



def analysis_char_weight(char_weight_filename, vocab_filename, new_weight_filename):
    """
    通过字典文件和字权重文件生成相应的权重文件，提供后续作为预训练权重
    :param char_weight_filename:
    :param vocab_filename:
    :param new_weight_filename: 保存新的权重的文件名
    :return:
    """
    weights = read_weight_filename(char_weight_filename)
    vocabs = read_vocab_filename(vocab_filename)
    vocab_weights = dict()  # 保存字和权重对
    for c in tqdm(vocabs):
        if c not in weights:
            w = random_weight(50)
            print("%s 不存在预训练权重" % c)
        else:
            w = weights[c]
        vocab_weights[c] = w

    # 保存文件
    with open(new_weight_filename, "w", encoding="utf-8") as f:
        for vocab, we in vocab_weights.items():
            we = list(map(str, we))
            tmp = vocab + "\t" + "\t".join(we)
            f.write(tmp + "\n")
    print("预训练词向量已保存！")


def statistic_one_post_num(fileholder, label_map={'1': "0", '2': '0', '3': "1", '4': "2"}):
    """
    统计不同post下面，各个类别的数目
    """
    # label_map = {'1': "0", '2': '0', '3': "1", '4': "2"}
    keys = list(set([v for k,v in label_map.items()]))
    keys.sort(reverse=False)
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    statistic_num = []
    for i, name in enumerate(tqdm(filenames)):
        df = read_excel(name)
        statistic_num.append({"name": name, "post_label": label_map[str(df["label"][0])]})
        tmp = dict()  # 记录各个类别的数目
        for l in df["label"][1:]:
            l = str(l)
            if l != '0':
                l = label_map[l]
                if l not in tmp:
                    tmp[l] = 0
                tmp[l] += 1
        statistic_num[-1]["labels"] = {k: str(v) for k, v in tmp.items()}

    with open("statistic.json", "w", encoding="utf-8") as f:
        json.dump(statistic_num, f, indent=4)

    with open("statistic.csv", "w", encoding="utf-8") as f:
        f.write("pid,post_label,label_" + ',label_'.join(keys)+ "\n")
        for item in statistic_num:
            f.write(item['name'].replace('data2\\','') + ",")
            f.write(item['post_label'] + ",")
            f.write(",".join([item['labels'][label] if label in item['labels'] else "0" for label in keys]) + "\n")


def analysis_sta(filename):
    with open(filename, "r", encoding="utf-8") as f:
        res = json.load(f)
        post_data = dict()
        for item in res:
            p, labels = item["post_label"], item["labels"]
            if p not in post_data:
                post_data[p] = dict()
            for k, v in labels.items():
                if k not in post_data[p]:
                    post_data[p][k] = 0
                post_data[p][k] += int(v)
        print('post_label,labels')
        pprint(post_data)


class analysis_users(object):
    def __init__(self, fileholder):
        filenames = read_dir(fileholder)
        # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
        dfs = []
        for name in tqdm(filenames):
            df = read_excel(name, "User")
            dfs.append(df)
        self.df = pd.concat(dfs, axis=0)
        self.df = self.df.drop_duplicates(subset='uid', keep='first', inplace=False).reset_index(drop=True)
        print(self.df.count())

    def analysis_gender(self):
        genders = self.df["gender"]
        return set(genders)


def excel2txt(fileholder, out_file):
    """
    将fileholder中的excel文件合起来转成txt
    """
    filenames = read_dir(fileholder)
    # filenames = [n for n in filenames if "dataAll.xlsx" not in n]
    dfs = []
    for i, name in tqdm(enumerate(filenames)):
        # if i == 10:
        #     break
        df = read_excel(name, "Comment")
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    content = df["content"]
    with open(out_file, "w", encoding="utf-8") as f:
        for c in content:
            # f.write(" ".join(str(c).replace('emotion]',']').replace('[emotion','[')) + "\n")
            f.write(re.sub(r'((\[emotion.*?emotion\])|(.))', r"\1 ", str(c)).replace('emotion]',']').replace('[emotion','[') + "\n")

    
if __name__ == "__main__":
    # 分析.xlsx后缀的文件，数一下各个图有多少个节点
    # analysis_data(fileholder="修正1.3/data2/", graph_file="graph.txt")  # graph.txt
    # 所有图的大小 排序输出
    # analysis_graph_file(graph_file="graph.txt", graph_file_res="graph_ana.txt")  # graph_ana.txt
    # 统计能构成图的评论数量
    # statistic_com_nums('graph_ana.txt', 2)

    # 统计 com 与 post 的长度情况 -> data_ana.txt
    statistic_lengths("修正1.3/data2/")  # data_ana.txt
    # 统计标签分布 和 label_0的比率 -> data_ana.txt
    # statistic_label("修正1.3/data2/")  # data_ana.txt

    # 生成词表 !!
    # get_vocab("修正1.3/data2/", "vocab.txt")  # vocab.txt
    # 保存预训练的字权重 !!
    # analysis_char_weight(char_weight_filename=r"embeddings/gigaword_chn.all.a2b.uni.ite50.vec",
    #                     vocab_filename="vocab.txt",
    #                     new_weight_filename="weights.txt")  # weights.txt
    # 将所有excel文件content合起来转成txt（空格分隔）
    # excel2txt(r"./data2", r"out.txt")  # out.txt

    # 统计每个post 下 所有com label 1-4 -> 0-2 的数量
    # statistic_one_post_num("修正1.3/data2/", {'1': "0", '2': '1', '3': "1", '4': "2"})  # statistic.json statistic.csv
    # 统计 post不同label下的 com label数量
    # analysis_sta("statistic.json")

    # 分析性别取值
    # tool = analysis_users('./修正1.3/data2/')
    # print(tool.analysis_gender())