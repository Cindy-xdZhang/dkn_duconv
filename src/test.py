# -*- encoding: utf-8 -*-
#'''
#@file_name    :test.py
#@description    :
#@time    :2020/02/21 16:25:18
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import torch
import os
import json
import time
from utils import *
import network 
from torch.nn import functional as F
from data_loader import My_dataset
from torch.utils.data import DataLoader
from evaluate import *
from main import build_models 
USE_CUDA = torch.cuda.is_available() 
class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
           return -1
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                #EOS_token则翻译成字符串
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['EOS'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences
    #如果在中途EOS会在addTopk翻译为文本，否则在最后用本函数翻译为文本
    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('EOS')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('EOS')
        return (words, self.avgScore())
def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(network.Global_device)
            # decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(1,batch_size) #[batch_size,1]
            # decoder_input = decoder_input.to(network.Global_device)
            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            top_value, top_index = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(top_index, top_value, decoder_hidden, beam_size, voc)
            #如果topk中有EOS才会terminal_sentences变长
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []
    #如果在中途EOS则会在addTopk翻译为文本，否则在最后用toWordScore函数翻译为文本
    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)
    return terminal_sentences[:1]
def topic_materialization(inputs_sentenses,data_dir,output_file_dir):
    """
    topic_materialization
    """
    topic_file=os.path.join(data_dir,"topic.test.txt")
    # inputs = [line.strip() for line in open(input_file, 'r')]
    inputs = inputs_sentenses
    topics = [line.strip() for line in open(topic_file, 'r',encoding="utf-8")]

    assert len(inputs) == len(topics)
    output_file=os.path.join(output_file_dir,"result.txt")
    if os.path.exists(output_file_dir)==False:os.mkdir(output_file_dir)
    fout = open(output_file, 'w',encoding="utf-8")
    for i, text in enumerate(inputs):
        topic_dict = json.loads(topics[i], encoding="utf-8")
        topic_list = sorted(topic_dict.items(), key=lambda item: len(item[1]), reverse=True)
        for key, value in topic_list:
            text = text.replace(key, value)
        fout.write(text + "\n")
    fout.close()
def eval(result_file, sample_file, eval_file):
    convert_result_for_eval(sample_file, result_file, eval_file)
    sents = []
    for line in open(eval_file):
        tk = line.strip().split("\t")
        if len(tk) < 2:
            continue
        pred_tokens = tk[0].strip().split(" ")
        gold_tokens = tk[1].strip().split(" ")
        sents.append([pred_tokens, gold_tokens])
        # calc f1
        f1 = calc_f1(sents)
        # calc bleu
        bleu1, bleu2 = calc_bleu(sents)
        # calc distinct
        distinct1, distinct2 = calc_distinct(sents)

        output_str = "F1: %.2f%%\n" % (f1 * 100)
        output_str += "BLEU1: %.3f%%\n" % bleu1
        output_str += "BLEU2: %.3f%%\n" % bleu2
        output_str += "DISTINCT1: %.3f%%\n" % distinct1
        output_str += "DISTINCT2: %.3f%%\n" % distinct2
        print(output_str)
def test_model(config):
    print('-Test:loading dataset...')
    pre_check_path=os.path.join(config.data_dir,"pre_check.txt")
    DuConv_test_DataSet=My_dataset("test",config.data_dir,config.voc_and_embedding_save_path)
    #test时batchsize=1
    test_loader = DataLoader(dataset=DuConv_test_DataSet,\
            shuffle=False, batch_size=1,drop_last=False,collate_fn=collate_fn)
    print('-Test:building models...')
    if config.continue_training != " " :
        checkpoint =torch.load(config.continue_training,map_location=network.Global_device) 
    else:
        raise Exception("No model exception when test model !")
    encoder,decoder=build_models(DuConv_test_DataSet.voc,config,checkpoint)
    print('-Initializing test process...')
    total_loss=0
    batch_size=config.batch_size
    voc=DuConv_test_DataSet.voc
    output_sentences=[]
    with torch.no_grad():
        if config.model_type=="gru":
            for batch_idx, data in enumerate(test_loader):
                print("testing: ",batch_idx,"/",len(test_loader)," ...")
                history,knowledge,responses=data["history"],data["knowledge"],data["response"]
                history,len_history,idx_unsort1 = padding_sort_transform(history)
                knowledge,len_knowledge,idx_unsort2 = padding_sort_transform(knowledge)
                responses,len_response,idx_unsort3 = padding_sort_transform(responses)
                if config.use_gpu and USE_CUDA: 
                    history,knowledge,responses,idx_unsort1,idx_unsort2,idx_unsort3 = history.cuda() ,\
                knowledge.cuda() ,responses.cuda(),idx_unsort1.cuda(),idx_unsort2.cuda(),idx_unsort3.cuda()
                unsort_idxs=(idx_unsort1,idx_unsort2)
                encoder_outputs, encoder_hidden = encoder(history,len_history,knowledge,len_knowledge,unsort_idxs)
                decoder_hidden = encoder_hidden[:decoder.n_layers]
                output_words_list=beam_decode(decoder,decoder_hidden,encoder_outputs,voc,config.beam_size,config.max_dec_len)
                output_words, score=output_words_list[0]
                output_sentence = ' '.join(output_words)
                output_sentences.append(output_sentence)
                with open(pre_check_path,'a') as f:
                    template='Re:\t'+output_sentence+"\n"
                    str=template.format
                    f.write(str)
        elif config.model_type=="trans":
            for batch_idx, data in enumerate(test_loader):
                print("testing: ",batch_idx,"/",len(test_loader)," ...")
                history,knowledge,responses=data["history"],data["knowledge"],data["response"]
                history = pad_sequence(history,batch_first=True, padding_value=0)
                knowledge = pad_sequence(knowledge,batch_first=True, padding_value=0)
                len_respons=[len(it) for it in responses]
                len_respons.sort()
                MAX_RESPONSE_LENGTH=len_respons[-1]-1
                if config.use_gpu and USE_CUDA: 
                    history,knowledge= history.cuda() ,knowledge.cuda()
                #enc_outs =[batchsize,seq,embedding]
                enc_output = encoder(history,knowledge)
                #transformer decoder input 是之前生成的所有句子-》注意观察一下POSITION ENCODING 的移位情况
                decoder_input = torch.LongTensor([SOS_token]).reshape(1,-1) #[batch_size,1]
                decoder_input = decoder_input.to(network.Global_device)
                for t in range(MAX_RESPONSE_LENGTH):
                    decoder_input = decoder_input.to(network.Global_device)
                    decoder_output, self_attentions, context_attentions = decoder( decoder_input, enc_output )
                    #TODO:我用的选最后一个加squeeze. [b,l,dim]原文中又是如何变成[b,dim]的？
                    decoder_output=decoder_output[:,-1,:].squeeze(1)
                    #topi为概率最大词汇的下标shape=[batch_Size=1,1]
                    _, topi = decoder_output.topk(1) # [batch_Size=1, 1]
                    decoder_input=torch.cat((decoder_input,topi),1)
                    #BATCHSIZE=1
                    if topi[0][0]==2:break
                decoder_input=decoder_input.squeeze(0).numpy().tolist()
                decoder_input_str=[voc.index2word[x] for x in decoder_input]
                output_sentence = ' '.join(decoder_input_str)
                with open(pre_check_path,'a') as f:
                    template='Re:\t'+output_sentence+"\n"
                    str=template.format
                    f.write(str)
                output_sentences.append(output_sentence)

    topic_materialization(output_sentences,config.data_dir,config.output_path)
    text_path=os.path.join(config.data_dir,"text.test_reduce.txt")
    eval_path=os.path.join(config.data_dir,"result_eval.txt")
    eval(config.output_path,text_path,eval_path)
def dev(handler):
    encoder,decoder,config,epoch=handler
    DuConv_test_DataSet=My_dataset("dev",config.data_dir,config.voc_and_embedding_save_path)
    dev_loader = DataLoader(dataset=DuConv_test_DataSet,\
            shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print('-Initializing Evaluate Process...')
    total_loss=0
    batch_size=config.batch_size
    voc=DuConv_test_DataSet.voc
    output_sentences=[]
    with torch.no_grad():
        if config.model_type=="gru":
            epoch_loss_avg=0
            for batch_idx, data in enumerate(dev_loader):
                history,knowledge,responses=data["history"],data["knowledge"],data["response"]
                #log2020.2.23:之前没有发现padding_sort_transform后每个batch内的顺序变了,必须把idx_unsort 也加进来
                history,len_history,idx_unsort1 = padding_sort_transform(history)
                knowledge,len_knowledge,idx_unsort2 = padding_sort_transform(knowledge)
                responses,len_response,idx_unsort3 = padding_sort_transform(responses)
                if config.use_gpu and USE_CUDA: 
                    history,knowledge,responses,idx_unsort1,idx_unsort2,idx_unsort3 = history.cuda() ,\
                knowledge.cuda() ,responses.cuda(),idx_unsort1.cuda(),idx_unsort2.cuda(),idx_unsort3.cuda()
                unsort_idxs=(idx_unsort1,idx_unsort2)
                encoder_outputs, encoder_hidden = encoder(history,len_history,knowledge,len_knowledge,unsort_idxs)
                decoder_hidden = encoder_hidden[:decoder.n_layers]
                loss=0
                MAX_RESPONSE_LENGTH=int(len_response[0].item())-1
                responses=responses.index_select(1,idx_unsort3)
                decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(1,batch_size) #[batch_size,1]
                decoder_input = decoder_input.to(network.Global_device)
                for t in range(MAX_RESPONSE_LENGTH):
                    decoder_output, decoder_hidden, decoder_attn = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                    )
                #topi为概率最大词汇的下标
                _, topi = decoder_output.topk(1) # [batch_Size, 1]

                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).reshape(1,batch_size)
                decoder_input = decoder_input.to(network.Global_device)  
                # decoder_output=[batch_Size, voc]  responses[seq,batchsize]
                loss += F.cross_entropy(decoder_output, responses[t+1], ignore_index=PAD_token)
            epoch_loss_avg+=loss.cpu().item() 
            epoch_loss_avg/=len(dev_loader)
            print('Evaluate Epoch: {}\t avg Loss: {:.6f}\ttime: {}'.format(
               epoch,epoch_loss_avg, time.asctime(time.localtime(time.time())) ))
            with open(config.logfile_path,'a') as f:
                template=' Evaluate Epoch: {}\tLoss: {:.6f}\ttime: {}\n'
                str=template.format(epoch,epoch_loss_avg,\
                    time.asctime(time.localtime(time.time())))
                f.write(str)
        elif config.model_type=="trans":
            epoch_loss_avg=0
            for batch_idx, data in enumerate(dev_loader):
                history,knowledge,responses=data["history"],data["knowledge"],data["response"]
                history = pad_sequence(history,batch_first=True, padding_value=0)
                knowledge = pad_sequence(knowledge,batch_first=True, padding_value=0)
                len_respons=[len(it) for it in responses]
                len_respons.sort()
                MAX_RESPONSE_LENGTH=len_respons[-1]-1
                responses =pad_sequence(responses,batch_first=True, padding_value=0)
                responses=responses.transpose(0,1)
                if config.use_gpu and USE_CUDA: 
                    history,knowledge= history.cuda() ,knowledge.cuda()
                #enc_outs =[batchsize,seq,embedding]
                enc_output = encoder(history,knowledge)
                #transformer decoder input 是之前生成的所有句子-》注意观察一下POSITION ENCODING 的移位情况
                decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(batch_size,1) #[batch_size,1]
                decoder_input = decoder_input.to(network.Global_device)
                loss=0
                for t in range(MAX_RESPONSE_LENGTH):
                    decoder_input = decoder_input.to(network.Global_device)
                    decoder_output, self_attentions, context_attentions = decoder( decoder_input, enc_output )
                    #TODO:我用的选最后一个加squeeze. [b,l,dim]原文中又是如何变成[b,dim]的？
                    decoder_output=decoder_output[:,-1,:].squeeze(1)
                    #topi为概率最大词汇的下标shape=[batch_Size=1,1]
                    _, topi = decoder_output.topk(1) # [batch_Size=1, 1]
                    decoder_input=torch.cat((decoder_input,topi),1)
                    loss += F.cross_entropy(decoder_output, responses[t+1], ignore_index=PAD_token)
                epoch_loss_avg+=loss.cpu().item() 
            epoch_loss_avg/=len(dev_loader)
            print('Evaluate Epoch: {}\t avg Loss: {:.6f}\ttime: {}'.format(
            epoch,epoch_loss_avg, time.asctime(time.localtime(time.time())) ))
            with open(config.logfile_path,'a') as f:
                template=' Evaluate Epoch: {}\t avg Loss: {:.6f}\ttime: {}\n'
                str=template.format(epoch,epoch_loss_avg,\
                    time.asctime(time.localtime(time.time())))
                f.write(str)
