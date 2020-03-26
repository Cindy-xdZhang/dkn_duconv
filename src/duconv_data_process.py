# -*- encoding: utf-8 -*-
#'''
#@file_name    :duconv_data_process.py
#@description    :
#@time    :2020/02/12 13:46:35
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''


import re
import os
import json
import collections
import random
"""
1. Dialogue Goal（goal）:
It contains two or three lines: the first contains the given dialogue path i.e., 
["Start", TOPIC_A, TOPIC_B].
 The other lines contains the relationship of TOPIC_A and TOPIC_B.
2. Knowledge：
Background knowledge related to TOPIC_A and TOPIC_B.
3. Conversation: 
4 to 8 turns of conversation.
4. Dialogue History: 
Conversation sequences before the current utterance, 
empty if the current utterance is at the start of the conversion.
5. Response: 
Gold response, which is only included in the test_1 set for development.
train: goal;knowledge;conversation;
test:goal; knowledge; conversation history; response
----------------------Idx-----------------------------------------------
DKN中，把kg.txt(rawID) 利用类似词汇表的方式，统计三元组、关系类型、实体的总数，
然后转成triple2id.txt(KGid triple+relationID),relation2id.txt(relationID),entity2id.txt(KGid2rawID); 
那么现在新的项目的ID设置为：
在train.txt 上建立词汇表，词汇到词汇ID，建立实体表称为Uniq_ID，实体2Uniq_ID.
triple2id.txt(Uniq_ID triple+relationID),relation2id.txt(relation2id),entity2id.txt(Uniq_ID2entity); 
test.txt也要建立词汇表
"""

#把多轮对话拆成多个单轮对话
#可单独使用，也被集成到了data_preprocess中
def convert_session_to_sample(session_file, sample_file):
    """
    convert_session_to_sample
    """
    fout = open(sample_file, 'w',encoding='utf-8')
    with open(session_file, 'r', encoding='utf-8') as f:
        #每一行也是一条独立的记录
        for i, line in enumerate(f):
            session = json.loads(line.strip(), encoding="utf-8", \
                                      object_pairs_hook=collections.OrderedDict)
            conversation = session["conversation"]

            for j in range(0, len(conversation), 2):
                sample = collections.OrderedDict()
                sample["goal"] = session["goal"]
                sample["knowledge"] = session["knowledge"]
                sample["history"] = conversation[:j]
                sample["response"] = conversation[j]

                sample = json.dumps(sample, ensure_ascii=False)

                fout.write(sample + "\n")

    fout.close()
#实现了data_preprocess, 整合sample成多轮对话和泛化步骤，
# tokenize也重新适应，另外我发现
# 1：tokenize最好到建立vocabulary时再用，text.txt中又泛化又tokenize，对话完全没意义了，因此目前的sample+泛化输出应该有数字
# ，下一步建立词汇表的时候再tokenize。（其实好像也无所谓，DUCONV自己跑出来的就是没有实际数字的对话。。）
#2.dkn 中泛化后文本不再以json形式存储，而是以一个长字符串加分隔符做全部数据。而我暂且保留了json格式，在json格式基础上实现泛化
def data_preprocess(path_raw,text_file,topic_file,topic_generalization=True,test=False,augment=True):
    # tokenize数字
    # tokenize<<xx>> 2020.0212
    # tokenize在sample、泛化后，在建立词汇表以前   2020.0202
    def tokenize(tokens):
        """
        tokenize
        print(tokenize("1999年5月"))-><num>年<num>月
        print(tokenize([["1999年5月","1929年5月","1991年3月"],["1999年5月","1929年5月","1991年3月"]]))
        """
        #整数、小数均替换为<num>
        if isinstance(tokens, str): 
            s = re.sub('\d+', ' <num> ', tokens)
            s = re.sub('\s<num>\s\.\s<num>\s', ' <num> ', s)
            s= re.sub('\《.+》',' <works> ',s)
            s=re.sub("[\.\!\/,$%^*()+\"\']+|[+——！。~@#￥%……&*（）-]", " ", s) 
            s=re.sub("\s{2,5}", " ", s) 
            return s
        elif isinstance(tokens, list):
            tokens_list = [tokenize(t) for t in tokens]
            return tokens_list
    #泛化topic的原子操作
    def generize(tokens,value, key):
        """
        generize
        """
        if isinstance(tokens, str): 
            s = tokens.replace(value, key)
            return s
        elif isinstance(tokens, list):
            tokens_list = [generize(t,value, key) for t in tokens]
            return tokens_list
    #采样和泛化topic
    def sample2multi_generize_topic(path_raw,text_file,topic_file):
        with open(path_raw, 'r', encoding='utf-8') as f:
            fout_text = open(text_file, 'w',encoding='utf-8')
            fout_topic = open(topic_file, 'w',encoding='utf-8')
            #每一行也是一条独立的记录
            if test==False:
                for i, line in enumerate(f):
                    session = json.loads(line.strip(), encoding="utf-8", \
                                                object_pairs_hook=collections.OrderedDict)
                    conversation = session["conversation"]

                    for j in range(0, len(conversation), 2):
                        sample = collections.OrderedDict()
                        sample["goal"] = session["goal"]
                        sample["knowledge"] = session["knowledge"]
                        sample["history"] = conversation[:j]
                        sample["response"] = conversation[j]

                        # response = sample["response"] if "response" in sample else "null"
                        
                        topic_a =  sample["goal"][0][1]
                        topic_b =  sample["goal"][0][2]
                        for i, [s, p, o] in enumerate(sample["knowledge"]):
                            if u"领域" == p:
                                if topic_a == s:
                                    domain_a = o
                                elif topic_b == s:
                                    domain_b = o

                        topic_dict = {}
                        if u"电影" == domain_a:
                            topic_dict["video_topic_a"] = topic_a
                        else:
                            topic_dict["person_topic_a"] = topic_a

                        if u"电影" == domain_b:
                            topic_dict["video_topic_b"] = topic_b
                        else:
                            topic_dict["person_topic_b"] = topic_b
                        if topic_generalization:
                            topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
                            for key, value in topic_list:
                                sample["goal"] = tokenize(generize(sample["goal"],value, key))
                                sample["knowledge"] = tokenize(generize(sample["knowledge"],value, key))
                                sample["history"] =  tokenize(generize(sample["history"],value, key))
                                sample["response"] =  tokenize(generize(sample["response"],value, key))
                                # model_text = model_text.replace(value, key)    
                        topic_dict = json.dumps(topic_dict, ensure_ascii=False)
                        model_text=json.dumps(sample, ensure_ascii=False)
                        fout_text.write(model_text + "\n")
                        fout_topic.write(topic_dict + "\n")
            else:
                for i, line in enumerate(f):
                    session = json.loads(line.strip(), encoding="utf-8", \
                                                object_pairs_hook=collections.OrderedDict)
                    sample = collections.OrderedDict()
                    sample["goal"] = session["goal"]
                    sample["knowledge"] = session["knowledge"]
                    sample["history"] = session["history"]
                    sample["response"] = session["response"]
                    topic_a =  sample["goal"][0][1]
                    topic_b =  sample["goal"][0][2]
                    for i, [s, p, o] in enumerate(sample["knowledge"]):
                        if u"领域" == p:
                            if topic_a == s:
                                domain_a = o
                            elif topic_b == s:
                                domain_b = o

                    topic_dict = {}
                    if u"电影" == domain_a:
                        topic_dict["video_topic_a"] = topic_a
                    else:
                        topic_dict["person_topic_a"] = topic_a

                    if u"电影" == domain_b:
                        topic_dict["video_topic_b"] = topic_b
                    else:
                        topic_dict["person_topic_b"] = topic_b
                    if topic_generalization:
                        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
                        for key, value in topic_list:
                            sample["goal"] = tokenize(generize(sample["goal"],value, key))
                            sample["knowledge"] = tokenize(generize(sample["knowledge"],value, key))
                            sample["history"] =  tokenize(generize(sample["history"],value, key))
                            sample["response"] =  tokenize(generize(sample["response"],value, key))
                            # model_text = model_text.replace(value, key)     
                    topic_dict = json.dumps(topic_dict, ensure_ascii=False)
                    model_text=json.dumps(sample, ensure_ascii=False)
                    fout_text.write(model_text + "\n")
                    fout_topic.write(topic_dict + "\n")


            fout_text.close()
            fout_topic.close()

    sample2multi_generize_topic(path_raw,text_file,topic_file)



if __name__ == "__main__":
    try:
        path_sample=os.path.join("dkn_duconv","duconv_data","train.txt")
        path_sample2=os.path.join("dkn_duconv","duconv_data","text.train.txt")
        path_sample3=os.path.join("dkn_duconv","duconv_data","topic.train.txt")
        data_preprocess(path_sample,path_sample2,path_sample3,test=False)
        path_sample=os.path.join("dkn_duconv","duconv_data","dev.txt")
        path_sample2=os.path.join("dkn_duconv","duconv_data","text.dev.txt")
        path_sample3=os.path.join("dkn_duconv","duconv_data","topic.dev.txt")
        data_preprocess(path_sample,path_sample2,path_sample3,test=False)
        path_sample=os.path.join("dkn_duconv","duconv_data","test_1.txt")
        path_sample2=os.path.join("dkn_duconv","duconv_data","text.test.txt")
        path_sample3=os.path.join("dkn_duconv","duconv_data","topic.test.txt")
        data_preprocess(path_sample,path_sample2,path_sample3,test=True)
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")