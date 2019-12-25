import re
import os
import numpy as np
import json
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
TODO:test.txt也要建立词汇表
"""
SOS_token = 0
EOS_token = 1
PAD_token = 2
MAX_TITLE_LENGTH=20
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}#word转索引，字典，word字符串做查询键值
        self.word2count = {}#word转频次计数器，似乎没卵用
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def encoding_sentence(self,sentence):
        """
        Encoding a sentence according to word2index map 
        :param sentence: a piece of news title
        :return: encodings of the sentence
        """
        array = sentence.split(' ')
        word_encoding = ['1'] * MAX_TITLE_LENGTH
        point = 0
        for s in array:
            if s in self.word2index:
                word_encoding[point] = str(self.word2index[s])
                point += 1
            if point == MAX_TITLE_LENGTH:
                break
        word_encoding = ','.join(word_encoding)#1728,732,5895,151,289,1224,1225,0,0,0
        return word_encoding
    def idx2sentence(self,sentence):
        array = sentence.split(',')
        word_encoding = ['EOS'] * MAX_TITLE_LENGTH
        point = 0
        for s in array:
            if int(s) in self.index2word:
                word_encoding[point] = str(self.index2word[int(s)])
                point += 1
            if point == MAX_TITLE_LENGTH:
                break
        word_encoding = ' '.join(word_encoding)
        return word_encoding
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
def parse_json_txtfile(files):
        transform_arrays=[]
        reader = open(files, encoding='utf-8')
        for line in reader:
            array= line.strip().split('\t')
            python_obj = json.loads(array[0])
            transform_arrays.append(python_obj)
        # print(transform_arrays[0]["goal"],"\n")
        # print(transform_arrays[0]["knowledge"],"\n")
        # print(transform_arrays[0]["conversation"],"\n")
        return transform_arrays
def word_index_embedding(train_data):
    conversations=[item["conversation"] for item in train_data]
    voc = Vocabulary("duconv_words")
    for conversation in conversations:
        for Isentence in conversation:
            voc.addSentence(Isentence)
    # print("Counted words in conversation:", voc.n_words)
    knowledges=[item["knowledge"] for item in train_data]
    for knowledge in knowledges:
        for Isentence in knowledge:
             for sentence in Isentence:
                voc.addSentence(sentence)      
    print("Counted words in conversations and knowledges:", voc.n_words)
    return voc
def entity_index_embedding(train_data):
    voc = KnowledgeList()
    knowledges=[item["knowledge"] for item in train_data]
    for knowledge in knowledges:
        for triple in knowledge:
                voc.addknowledge(triple)      
    print("Counted triples:", voc.triple_cnt,"\nCounted entities",voc.entity_cnt)
    return voc
def _check_KnowledgeList_and_Vocabulary_implementation():
    path=os.path.join("data","duconv","train.txt")
    train_data=parse_json_txtfile(path)     
    voc=word_index_embedding(train_data)
    knowledge=entity_index_embedding(train_data)
    for _ in range(5):
        idx=random.randint(1,knowledge.triple_cnt)
        a=knowledge.tripleList[idx]
        print(knowledge.index2entity[a[0]],"r:",knowledge.index2relation[a[1]],"t:",knowledge.index2entity[a[2]])

    idx=random.randint(1,  train_data.__len__())
    conversation=train_data[idx]["conversation"]
    print(conversation)   
    int_conv=[voc.encoding_sentence(x) for x in conversation]
    print(int_conv)
    recover_conv=[voc.idx2sentence(x)  for x in int_conv ]
    print(recover_conv)
def prepare_for_transX(data,triple_out='triple2id.txt', relation_out='relation2id.txt', entity_out='entity2id.txt'):
    writer_triple = open(triple_out, 'w', encoding='utf-8')
    writer_relation = open(relation_out, 'w', encoding='utf-8')
    writer_entity = open(entity_out, 'w', encoding='utf-8')
    print('writing triples to triple2id.txt ...')
    writer_triple.write('%d\n' % data.triple_cnt)
    for idx in range(data.triple_cnt):
        head_index, tail_index, relation_index=data.tripleList[idx][0],\
            data.tripleList[idx][2],data.tripleList[idx][1]
        writer_triple.write(
            '%d\t%d\t%d\n' % (head_index, tail_index, relation_index))#UniqID triple

    print('writing relations to relation2id.txt ...')
    writer_relation.write('%d\n' % data.relation_cnt)
    for i, relation in enumerate(data.relation_list):
        writer_relation.write('%s\t%d\n' % (relation, i))#relation2id
    print('relation size: %d' % data.relation_cnt) 

    print('writing entities to entity2id.txt ...')
    writer_entity.write('%d\n' % data.entity_cnt)
    for i, entity in enumerate(data.entity_list):
        writer_entity.write('%s\t%d\n' % (entity, i))#relation2id
    print('entity size: %d' % data.entity_cnt)


path=os.path.join("data","duconv","train.txt")
train_data=parse_json_txtfile(path)     
# # voc=word_index_embedding(train_data)
knowledge=entity_index_embedding(train_data)
prepare_for_transX(knowledge)
   


 
    
