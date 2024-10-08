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

# import win32file
from tqdm import tqdm
from queue import Queue
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, cpu_count
import numpy as np
import re
from pprint import pprint
from utils.clean_data import clean_string, cut_sentence



deal_num = 0




def read_dir(fileholder):
    
    fileholder = os.path.join(fileholder, "*.xlsx")
    filenames = glob.glob(fileholder)
    filenames = [f for f in filenames if f != "dataAll.xlsx"]
    return filenames


class DataProcess(object):
    def __init__(self, mode, isMul=False):
        self.label_map = {'0': "-1", '1': "0", "2": "0", '3': "1", "4": "2"}  
        self.user_tool = UserDataProcessor()
        self.mode = mode
        self.isMul = isMul
        #self.all = 0
        #self.start = []
        #self.finish = []

    def run(self, filenames, maxMul):
        
        # filenames = self.read_dir(self.fileholder)
        print("data_process", self.mode)
        # filenames = [n for n in filenames if "dataAll.xlsx" not in n][:10]
        filenames = [n for n in filenames if "dataAll.xlsx" not in n]
        # 加入多进程
        if self.isMul:
            self.all = len(filenames)
            processes = []
            with ProcessPoolExecutor(cpu_count() if cpu_count() <= maxMul else maxMul) as pool:
                for filename in tqdm(filenames):
                    processes.append(pool.submit(self.analysis_data_one_file, filename))

            graph_res = []
            for process in processes:
                graph_res.extend(process.result())
        else:
            graph_res = []
            for filename in tqdm(filenames):
                graph_res.extend(self.analysis_data_one_file(filename))


        res = []
        for o in tqdm(graph_res):
            res.append(self.deal_one_graph(o))
        return res

    def deal_one_graph(self, datas):
        
        datas = sorted(datas, key=lambda x: -len(cut_sentence(clean_string(x["content"]))))

        for data in datas:
            if not isinstance(data["content"], str):  
                data["content"] = ""
        adj = self.get_adjcent_matrix(datas)
        res = {
            "cid": np.array([data["cid"] for data in datas]),
            "content": np.array([data["content"] for data in datas]),
            "label": np.array([int(self.label_map[str(data["label"])]) for data in datas]),
            "adj": adj,
            "user_infos": np.array([data["user_infos"] for data in datas])
        }
        return res

    def get_adjcent_matrix(self, datas):

        node2id = dict()
        for i, data in enumerate(datas):
            cid = data["cid"]
            node2id[cid] = i

        adj = np.eye(len(datas), len(datas))
        for i, data in enumerate(datas):
            cid, fNode = data["cid"], data["fNode"]
            if fNode not in node2id:
                continue
            j = node2id[fNode]
            if i != -1 and j != -1:
                adj[i][j] = 1
                adj[j][i] = 1

        return adj

    def read_excel(self, filename, sheet_name=0):
        df = pd.read_excel(filename, sheet_name=sheet_name)
        return df

    def recursive(self, links):

        traversed = set()
        graph_num = 0
        graphs = []

        for node in links:
            if node not in traversed:  
                tmp_nodes = []  
                graph_num += 1
                q = Queue()
                q.put(node)  
                while q.qsize():  
                    q_size = q.qsize()
                    
                    for _ in range(q_size):
                        n = q.get()
                        if n not in traversed:
                            tmp_nodes.append(n)
                            traversed.add(n)
                            
                            for link_n in links[n]:
                                q.put(link_n)
                tmp_nodes = list(set(tmp_nodes))
                # tmp_nodes.sort()
                graphs.append(tmp_nodes)

        return graphs

    def statistic_graph(self, df):
        cids = df["cid"] 
        fNodes = df["fNode"] 
        assert len(cids) == len(fNodes) 

        labels = dict()
        for cid, l in zip(df["cid"], df["label"]):
            labels[str(cid)] = self.label_map[str(l)]

        root = str(cids[0]) 
        root = root + "[sep]" + labels[root]  
        links = dict()  
        for i in range(1, len(df)): 
            if str(df["content"][i]) == "nan":
                continue
            cid, fNode = str(cids[i]), str(fNodes[i])
            cid = cid + "[sep]" + labels[cid]
            fNode = fNode + "[sep]" + labels[fNode]
            if cid not in links:
                links[cid] = set()
            if fNode != root:
                if fNode not in links:
                    links[fNode] = set()
                links[cid].add(fNode)
                links[fNode].add(cid)

        graphs = self.recursive(links)
        graphs = [[root, ] + g for g in graphs]
        return graphs

    def analysis_data_one_file(self, filename):
        print('start:', filename)
        df = self.read_excel(filename, sheet_name="Comment")
        user_df = self.read_excel(filename, sheet_name="User")
        graphs = self.statistic_graph(df)

        df_dict = dict()
        for i in range(0, len(df)):
            cid, fNode, content, label = str(df["cid"][i]), str(df["fNode"][i]), str(df["content"][i]), str(df["label"][i])
            if i == 0:
                fNode = "-1"
            df_dict[cid] = {
                "cid": cid, "fNode": fNode, "content": content, "label": label, "user_infos": self._deal_user_info(user_df, i, self.user_tool)
            }

        res = []
        for graph in graphs:
            count_0 = 0
            one_item = []
            for node in graph:
                n, l = node.split("[sep]")
                if l == "-1":
                    count_0 += 1
                one_item.append(df_dict[n])
            if self.mode == "predict":
                res.append(one_item)
            else:
                if count_0 != len(graph):
                    res.append(one_item)
        print('finish:', filename)
        #self.finish.append(filename)
        #print('start:', len(self.start), 'finish:', len(self.finish))
        return res

    def _deal_user_info(self, user_df, idx, user_tool):
        shuxing = ["gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place", "verify_info", "coupon"]
        res = { t: user_df[t][idx] for t in shuxing}
        if str(res["verify_info"]) == "nan":
            verify_info = ""
        else:
            verify_info = res["verify_info"]
        if str(res["coupon"]) == "nan":
            coupon = ""
        else:
            coupon = res["coupon"]
        res["verify_info_coupon"] = re.sub( r"[ ]+", " ", verify_info + "。" + coupon)
        del res[shuxing[-2]]
        del res[shuxing[-1]]
        return user_tool.run(res)


class UserDataProcessor(object):

    def __init__(self):
        self.sparse_shuxing = ["gender", "focus_num", "fans_num", "blogs_num", "verify", "vip", "edu", "place"]

        if os.environ.get("_env", None) == "dev":
            filename = r"edu.xlsx"
        else:
            filename = r"edu.xlsx"
        df = pd.read_excel(filename)
        self.school2edu = {k: v for k, v in zip(df["edu"], df["class"])}

    def run(self, user_infos):
        for k in self.sparse_shuxing:
            func = getattr(self, "_deal_" + k)
            user_infos[k] = func(user_infos[k])
        return user_infos

    def _deal_gender(self, inputs):
        if str(inputs) == "nan":
            return 0
        return int(inputs)

    def _deal_focus_num(self, inputs):
        if str(inputs) == "nan":
            return 0
        spans = [
            [0, 10], [11, 50], [51, 200], [201, 500], [501, 1000], [1001, 2000], [2001, 5000], [5001, 10000], [10001, 1e12]
        ]
        for idx, (s, e) in enumerate(spans):
            if s <= inputs <= e:
                return idx + 1
        return 0

    def _deal_fans_num(self, inputs):
        if str(inputs) == "nan":
            return 0
        spans = [
            [0, 10], [11, 50], [51, 200], [201, 500], [501, 1000], [1001, 2000], [2001, 5000], [5001, 10000],
            [1e4+1, 1e5], [1e5+1, 1e6], [1e6+1, 1e7], [1e7+1, 1e8], [1e8+1, 1e9], [1e9+1, 1e12]
        ]
        for idx, (s, e) in enumerate(spans):
            if s <= inputs <= e:
                return idx + 1
        return 0

    def _deal_blogs_num(self, inputs):
        return self._deal_fans_num(inputs)

    def _deal_verify(self, inputs):
        verify_dict = {
            "无认证": 0,"微博个人认证": 1, "微博官方认证": 2
        }
        return verify_dict.get(inputs, 0)

    def _deal_vip(self, inputs):
        if 0 <= inputs <= 7:
            return int(inputs)
        return 0

    def _deal_edu(self, inputs):
        return self.school2edu.get(inputs, -1) + 1

    def _deal_place(self, inputs):
        place_dict = {
            '其他': 0, '海外': 1, '上海': 2, '北京': 3, '广东': 4, '澳门': 5, '浙江': 6, '江苏': 7,
            '福建': 8, '重庆': 9, '湖北': 10, '辽宁': 11, '山东': 12, '广西': 13, '黑龙江': 14,
            '四川': 15, '河北': 16, '湖南': 17, '甘肃': 18, '吉林': 19, '天津': 20, '贵州': 21,
            '新疆': 22, '青海': 23, '海南': 24, '安徽': 25, '云南': 26, '陕西': 27, '河南': 28,
            '山西': 29, '香港': 30, '西藏': 31, '江西': 32, '宁夏': 33, '内蒙古': 34, '台湾': 35}
        inputs = str(inputs)
        inputs = inputs.split(" ")[0]
        return place_dict.get(inputs, 0)


def analysis_graph_file(graph_file):
    """
    read graph_file data
    :param graph_file: graph file
    :return:
    """
    nums = []
    with open(graph_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            nums.extend(list(map(int, line[1:])))
    nums.sort(reverse=True)
    print(111)



if __name__ == "__main__":
    # analysis_data(fileholder="data2/", graph_file="graph.txt")
    # analysis_graph_file(graph_file="graph.txt")

    data_processor = DataProcess()
    filenames = read_dir(r"D:\111\2022-05-03-graph\data2")
    data_processor.run(filenames)
