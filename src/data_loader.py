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
    def __init__(self,  mode="train",dir="C:\\Users\\10718\\PycharmProjects\\dkn_duconv\\duconv_data",voc_save_path="dkn_duconv/"):
        self.mode = mode
        self.dir = dir
        self.data,self.voc=self.build_corpus_data(voc_save_path)
    #学习DUCONV把 每个item多个konwledge三元组 join成空格分隔的一个长句子
    def build_corpus_data(self,voc_save_path):
        text_path1=os.path.join(self.dir,"text."+"train"+".txt")
        text_path2=os.path.join(self.dir,"text."+"dev"+".txt")
        text_path3=os.path.join(self.dir,"text."+"test"+".txt") 
        train_data=parse_json_txtfile(text_path1)
        dev_data=parse_json_txtfile(text_path2)
        test_data=parse_json_txtfile(text_path3)
        all_data=train_data+dev_data+test_data
        # voc=word_index(train_data,voc_save_path+"1")# 54941
        voc=word_index(all_data,voc_save_path)#62008
        if self.mode=="train":
            data=train_data
        elif self.mode=="dev":
            data=dev_data
        elif self.mode=="test":   
            data=test_data
        else:
            raise ValueError("Working mode in My_dataset has to be train/dev/test.")
        for item in data:
            item.pop('goal')
        #对response,history,knowkedge idx化但不padding，
        # 后续进网络每个batch交给torch.nn.utils.rnn.pad_sequence 进行padding
        voc.idx_corpus(data)
        return data,voc
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
"""
v0.0.1:
输入数据的item格式：
{
'knowledge': [24, 895, 24, 294, 24107, 351, 34158, 24, .....]; 
超长的知识合成为一句的的int idx列表，每个item的长度不等->后续进网络每个batch padding packed 知识之间以0分割
'history': [1，132, 71, 133, 134, 28, 29, 2, 1, ..]; 句子以1开头，2结束分割
多句对话历史合成为一句长句 int idx列表
'response': [24, 137, 138, 101, 134, 96, 19, 106, 139, 104, 29, 2, 0, 0, 0, 0, 0, 0, 0, 0]
每个item的response长度相同为MAX_TITLE_LENGTH
}
输入数据均为变长数据-》交给torch.padsequence等后续padding

"""
# x=My_dataset()
# for i in range(10):
#     print(x[i])
#以下代码测试torch.nn.utils.rnn.pack_padded_sequence表明变成序列不能长度为0
# RuntimeError: Length of all samples has to be greater than 0, but found an element in 'lengths' that is <= 0
# from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
# input_sEQ=[torch.LongTensor(it) for it in [[], [13069,615, 833, 238, 19, 872, 133, 22, 174, 43, 29, 2],[34, 29, 2, 73]] ]
# input_len=torch.Tensor([len(it) for it in input_sEQ])
# input_sEQ=pad_sequence(input_sEQ,batch_first=True, padding_value=0)
# _,idx_sort=torch.sort(input_len,dim=0,descending=True)
# input_sEQ=input_sEQ.index_select(0,idx_sort)
# lengths=list(input_len[idx_sort])
# input_kg_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_sEQ, lengths,batch_first=True)
