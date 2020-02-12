# -*- encoding: utf-8 -*-
#'''
#@file_name    :data_loader.py
#@description    :
#@time    :2020/02/12 13:46:41
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''

import torch.utils.data.dataset
import numpy as np
from utils import *

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
class My_dataset(Dataset):
    # root 是训练集的根目录， mode可选的参数是train，test，validation，分别读取相应的文件夹
    def __init__(self,  mode,dir="C:\\Users\\10718\\PycharmProjects\\data\\duconv\\"):
        self.mode = mode
        self.dir = dir
        #元素为字典的列表
        self.data,self.voc=self.build_corpus_data()
    #学习DUCONV把 每个item多个konwledge三元组 join成空格分隔的一个长句子
    def build_corpus_data(self):
        text_path=self.dir+"text."+self.mode+".txt"
        train_data=parse_json_txtfile(text_path)
        voc=word_index(train_data)
        for item in train_data:
            item["knowledge"]=' '.join([' '.join(spo) for spo in item["knowledge"]])
            item.pop('goal')
        # voc.idx_padding_corpus(train_data)
        return train_data,voc
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
# myset=My_dataset("train")
# for _ in range(5):
#     idx=random.randint(1,myset.__len__())
#     print(myset[idx])
# print(myset[idx]["knowledge"],len(myset[idx]["knowledge"]))
# print(myset[idx]["response"],len(myset[idx]["response"]))
#TODO:知识该如何输入?知识合并成一句话，长度远超于一般句子