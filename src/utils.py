# -*- encoding: utf-8 -*-
#'''
#@file_name    :utils.py
#@description    :
#@time    :2020/02/12 13:46:49
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import time
import os
import json
import random
import numpy as np
import torch 
from torch import load,save
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
#Vocabulary中的token标号
PAD_token = 0
SOS_token = 1
EOS_token = 2
#id2str or str2id串时的最大长度，也等同于padding的最大长度
#average # utterances per dialog	9.1
#average # words per utterance	10.6
MAX_TITLE_LENGTH=20
WORD_EMBEDDING_DIM_PRETRAIN=20

class Vocabulary:
    def __init__(self, name="None"):
        self.name = name
        self.word2index = {"PAD":0 ,"SOS": 1, "EOS":2 }#word转索引，字典，word字符串做查询键值
        # self.word2count = {}#word转频次计数器，似乎没卵用
        # self.index2word = {0: "PAD", 1: "SOS", 2:"EOS"}
        self.n_words = 3  # Count SOS and EOS
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index and word != '':
            self.word2index[word] = self.n_words
            # self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        # else:
            # self.word2count[word] += 1
    def idx_corpus(self,train_data):
        """ 数字化 and padding every sentense in corpus 
            输入  "1, 23 ,45 ,56"格式字符串
            输出  [ 1, 23 ,45 ,56,1,0,0,0,0]int 列表->直接进网络
            #log2020.2,15:原本在此处idx化同时max_length指定为MAX_TITLE_LENGTH，我发现对response还行，
            #毕竟每一项都只是一句话，长度差距不大。对于知识虽然每一项长度差距更大，但还勉强可以定另一个100左右的max length，虽然感觉就不好了；
            #而对于history，人为设定合并成一句后的maxlength很不好，因为每份数据的history可能是0 和50-100 words，人为指定长度padding
            #很不好。干脆三项数据都保留变长，进去一个batch一起根据max_length padding 然后用pack
        """
        def merge_history(str_in):
            if len(str_in)>=1:
                res=""
                for sentense in str_in:
                    res=res+"SOS "+sentense+' EOS '
                item_cp=[]
                for point in res.split(" "):
                    if point !="":
                        item_cp.append(self.word2index[point]) 
                return item_cp
            else:
                str_in.append(1)
                str_in.append(2)
                return str_in
        for item in train_data:
            #history
            item["history"]=merge_history(item["history"])
            #response
            item["response"]="SOS "+item["response"]+" EOS"
            item_cp=[]
            for point in item["response"].split(" "):
                if point !="":
                    item_cp.append(self.word2index[point])
            item["response"]  =  item_cp    
            #kg
            kg=' PAD '.join([' '.join(spo) for spo in item["knowledge"]])
            length=len(kg.split())
            item["knowledge"]=self.encoding_sentence(kg,max_length=length)
            
            


    def encoding_sentence(self,sentence,max_length=MAX_TITLE_LENGTH,form="int"):
        """
        Encoding a sentence according to word2index map 
        :param sentence: a piece of news title
        :return: encodings of the sentence
        """
        if isinstance(sentence, str): 
            array = sentence.split(' ')
            word_encoding = [str(PAD_token)] * max_length 
            point = 0
            for s in array:
                #未知的字符也会用pad替代
                if s in self.word2index:
                    word_encoding[point] = str(self.word2index[s]) 
                    point += 1
                if point == max_length-1:
                    break
            word_encoding[point]=str(EOS_token) 
            if form == "str": word_encoding = ','.join(word_encoding)#1728,732,5895,151,289,1224,1225,0,0,0
            elif form =="int":word_encoding=[int(item) for item in word_encoding]
            return word_encoding
        elif isinstance(sentence, list): 
            tokens_list = [self.encoding_sentence(t,max_length=max_length,form=form) for t in sentence]
            return tokens_list
    def idx2sentence(self,sentence,no_padding=False,max_length=MAX_TITLE_LENGTH,):
        if isinstance(sentence, str):
            if no_padding ==False: 
                array = sentence.split(',')
                word_encoding = ['EOS'] * max_length
                point = 0
                for s in array:
                    if int(s) in self.index2word:
                        word_encoding[point] = str(self.index2word[int(s)])
                        point += 1
                    if point == max_length:
                        break
                word_encoding = ' '.join(word_encoding)
                return word_encoding
            else:return [str(self.index2word[int(ix)]) for ix in sentence]
        elif isinstance(sentence, list): 
            tokens_list = [self.idx2sentence(t,max_length,no_padding) for t in sentence]
            return tokens_list
class KnowledgeList:#UniqID
    def __init__(self):
        self.entity2index = {}#word转索引，字典，word字符串做查询键值
        self.index2entity = {}
        self.entity_list=[]
        self.entity_cnt = 0 

        self.relation2index = {}#word转索引，字典，word字符串做查询键值
        self.index2relation = {}
        self.relation_cnt = 0 
        self.relation_list=[]

        self.tripleList=[]
        self.triple_cnt = 0 

    def addknowledge(self, triple):
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head in self.entity2index:
            head_index = self.entity2index[head]
        else:
            head_index = self.entity_cnt
            self.entity2index[head] = self.entity_cnt
            self.index2entity[self.entity_cnt]=head
            self.entity_list.append(head)
            self.entity_cnt += 1
        if tail in self.entity2index:
            tail_index = self.entity2index[tail]
        else:
            tail_index = self.entity_cnt
            self.entity2index[tail] = self.entity_cnt
            self.index2entity[self.entity_cnt]=tail
            self.entity_list.append(tail)
            self.entity_cnt += 1
        if relation in self.relation2index:
            relation_index = self.relation2index[relation]
        else:
            relation_index = self.relation_cnt
            self.relation2index[relation] = self.relation_cnt
            self.index2relation[self.relation_cnt] = relation
            self.relation_list.append(relation)
            self.relation_cnt += 1
        triple=[head_index,relation_index,tail_index]#UniqID
        self.tripleList.append(triple)
        self.triple_cnt+=1  
#读取json文件
def parse_json_txtfile(files):
        transform_arrays=[]
        reader = open(files, encoding='utf-8')
        for line in reader:
            array= line.strip().split('\t')
            python_obj = json.loads(array[0])
            transform_arrays.append(python_obj)
        return transform_arrays
#word_index
##record 2020.0202: 
#将本代码变成通用的，既可用于raw corpus也可用于sample 后
#record 2020.0203: 
# 这个代码实现现在无问题，之所以原版本出现raw corpus和 sample 后字数差好几百，是因为中间的response只是句子不是列表却被当成了列表，
# 然后split就成了每个字一个词了。比如之前是 2019年10月 变成了  2 0 1 9 年 1 0 月 多个word 。此问题已修正
#另外raw数据sample后部分句子会被删除（response只取第1 3 5 。。。句，因此比如总共只有四句则第四局会被省略），
# 因此sample后还是会出现word count下降
def word_index(train_data,voc_save_dir):
    voc_save_path=os.path.join(voc_save_dir, '{!s}.tar'.format('duconv_voc'))
    if os.path.exists(voc_save_path)==False:
        print("-building voc ....")
        #for raw corpus  item["conversation"]
        if "conversation" in train_data[0].keys():
            conversations=[item["conversation"] for item in train_data]
        #for samples item["response"]/["history"]
        else:
            conversations=[item["history"] for item in train_data]
            response=[[item["response"]] for item in train_data]
            conversations=conversations+response
        voc = Vocabulary("duconv_words")
        for conversation in conversations:
            for Isentence in conversation:
                voc.addSentence(Isentence)
        # print("Counted words in conversation:", voc.n_words)
        knowledges=[item["knowledge"] for item in train_data]
        for knowledge in knowledges:
            kg=' '.join([' '.join(spo) for spo in knowledge]).split()
            for sentence in kg:
                voc.addSentence(sentence)
        print("-Counted words in conversations and knowledges:", voc.n_words)
        if  os.path.exists(voc_save_dir)==False:os.mkdir(voc_save_dir)
        save(voc,voc_save_path)
    else:
        print("-Loading voc from file....")
        voc=load(voc_save_path)
    return voc
#entity_index(Uniq_ID)
def entity_index(train_data):
    voc = KnowledgeList()
    knowledges=[item["knowledge"] for item in train_data]
    for knowledge in knowledges:
        for triple in knowledge:
                voc.addknowledge(triple)      
    print("-Counted triples:", voc.triple_cnt,"\nCounted entities",voc.entity_cnt)
    return voc
#检查

def _check_KnowledgeList_and_Vocabulary_implementation():
    # path_raw=os.path.join("data","duconv","train.txt")
    # path_sample=os.path.join("data","duconv","sample.train.txt")
    #convert_session_to_sample(path_raw,path_sample)
    path_sample2=os.path.join("dkn_duconv","duconv_data","text.train.txt")
    text_path2=os.path.join("dkn_duconv","duconv_data","text.test.txt")
    text_path3=os.path.join("dkn_duconv","duconv_data","text.dev.txt")
    # data_preprocess(path_sample,path_sample2,path_sample3)
    train_data=parse_json_txtfile(path_sample2)  
    dev_data=parse_json_txtfile(text_path2)
    test_data=parse_json_txtfile(text_path3)  
    all_data=train_data+dev_data+test_data 
    voc=word_index(all_data,"./dkn_duconv")
    
    build_embedding(voc)

    # knowledge=entity_index(train_data)
    # for _ in range(5):
    #     idx=random.randint(1,knowledge.triple_cnt)
    #     a=knowledge.tripleList[idx]
    #     print(knowledge.index2entity[a[0]],"r:",knowledge.index2relation[a[1]],"t:",knowledge.index2entity[a[2]])
    # for _ in range(2):
    #     idx=random.randint(1,  train_data.__len__())
    #     conversation=train_data[idx]["history"]
    #     print("history:\n",conversation)   
    #     int_conv=voc.encoding_sentence(conversation) 
    #     print("index-history:\n",int_conv)
    #     recover_conv=voc.idx2sentence(int_conv)  
    #     print("recover-history:\n",recover_conv)
    #     print("response:\n")
    #     response=train_data[idx]["response"]
    #     print(response)   
    #     int_conv=voc.encoding_sentence(response)
    #     print(int_conv)
    #     recover_conv= voc.idx2sentence(int_conv)  
    #     print(recover_conv)
def PCA_embedding(path_in,path_out):
    path_sample2=os.path.join("dkn_duconv","duconv_data","text.train.txt")
    text_path2=os.path.join("dkn_duconv","duconv_data","text.test.txt")
    text_path3=os.path.join("dkn_duconv","duconv_data","text.dev.txt")
    train_data=parse_json_txtfile(path_sample2)  
    dev_data=parse_json_txtfile(text_path2)
    test_data=parse_json_txtfile(text_path3)  
    all_data=train_data+dev_data+test_data 
    voc=word_index(all_data,"./dkn_duconv")
    Words=[]
    vecs=[]
    with open(path_in, 'r', encoding='utf-8') as f:
        # embeded_words=0
        word2index=voc.word2index
        for i, line in enumerate(f):
            if i==0:continue
            word,vec=(line.strip()).split(" ", 1)
            if word in word2index.keys():
                Words.append(word)
                num_vec= np.array([float(item) for item in vec.split()])
                vecs.append(num_vec)
        f.close()
    pca = PCA(n_components=WORD_EMBEDDING_DIM_PRETRAIN)   #降到2维
    pca.fit(vecs)
    newX=pca.fit_transform(vecs)
    with open(path_out,'w',encoding='utf-8') as f:
        for idx,word in enumerate(Words):
            newp=[str(num) for num in newX[idx]]
            str_vec=" ".join(newp)
            template=str(word)+"\t"+str_vec+"\n"
            f.write(template)
        f.close()
def build_embedding(Vocabulary=None,voc_embedding_save_dir="dkn_duconv"):
    def load_pretrain_SGNS(embeddings):
        embeddings_file=os.path.join(voc_embedding_save_dir,"sgns.wiki_pca.txt")
        embeded_words=0
        word2index=Vocabulary.word2index
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word,vec=(line.strip()).split("\t", 1)
                if word in word2index.keys():
                    num_vec= np.array([float(item) for item in vec.split()])
                    embeddings[word2index[word]]=num_vec
                    embeded_words+=1
        print("Using sgns Word2Vec："+str(float(embeded_words/Vocabulary.n_words)*100)+ "% words are embedded.")
    if Vocabulary==None:raise Exception("No vocabulary information available!")
    else:
        voc_embedding_save_path=os.path.join(voc_embedding_save_dir, '{!s}.npy'.format('duconv_voc_embedding_'+str(Vocabulary.n_words)+"_"+str(WORD_EMBEDDING_DIM_PRETRAIN)))
        if os.path.exists(voc_embedding_save_path)==False:
            word2index=Vocabulary.word2index
            print('-getting word embeddings of '+ str(Vocabulary.n_words)  +' words from pretrain model...')
            embeddings = np.random.rand(len(word2index), WORD_EMBEDDING_DIM_PRETRAIN)
            load_pretrain_SGNS(embeddings)
            embeddings[0]=0
            print('- writing word embeddings ...')
            if  os.path.exists(voc_embedding_save_dir)==False:os.mkdir(voc_embedding_save_dir)
            np.save(voc_embedding_save_path, embeddings)
            return embeddings
        elif os.path.exists(voc_embedding_save_path)==True:
            print('-load word embeddings ...')
            embeddings=np.array(np.load(voc_embedding_save_path))
            return embeddings
        else:
            raise Exception("Unknown Exception when build_embedding !")

def str2bool(v):
    """ str2bool """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def record_train_step(logfile_path,message,overall_loss=None):
    if overall_loss is None:
        epoch,batch_idx , epoch_length,print_loss_avg=message
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
                epoch, batch_idx , epoch_length,
                100. * batch_idx / epoch_length, print_loss_avg, time.asctime(time.localtime(time.time())) ))
        with open(logfile_path,'a') as f:
            template=' Train Epoch: {} [{}/{}]\tLoss: {:.6f}\ttime: {}\n'
            str=template.format(epoch,batch_idx ,epoch_length,print_loss_avg,\
            time.asctime(time.localtime(time.time())))
            f.write(str)
    else: 
        epoch=message
        with open(logfile_path,'a') as f:
            template=' Train Epoch: {} \t Overall Loss: {:.6f}\t time: {}\n'
            str=template.format(epoch, overall_loss,time.asctime(time.localtime(time.time())))
            print(str)
            f.write(str)

def padding_sort_transform(input_sEQ):
    """ input:[B x L]batch size 个变长句子TENSOR
    返回[L*B ]个定长句子TENSOR
    """
    input_len=torch.Tensor([len(it) for it in input_sEQ])
    input_sEQ=pad_sequence(input_sEQ,batch_first=True, padding_value=0)
    _,idx_sort=torch.sort(input_len,dim=0,descending=True)
    _, idx_unsort= torch.sort(idx_sort, dim=0)
    input_sEQ=input_sEQ.index_select(0,idx_sort)
    lengths=list(input_len[idx_sort])
    # batch first 只影响output 不影响hidden的形状 所以batch first=false格式更统一,因此此处转置
    input_sEQ=input_sEQ.transpose(0, 1)
    return input_sEQ,lengths,idx_unsort
def collate_fn(batch):
    history = [torch.LongTensor(item['history'] )for item in batch]
    knowledge = [torch.LongTensor(item['knowledge']) for item in batch]
    response = [torch.LongTensor(item['response']) for item in batch]
    return {
        'history': history,
        'knowledge': knowledge,
        'response': response,
    }
def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images    
# _check_KnowledgeList_and_Vocabulary_implementation()

# import torch
# x = torch.rand(2,20)
# y = torch.split(x,[2,5,10,3],dim=1) 
# print(x)
# print(y)
