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
from utils import *
from main import build_models
import network 
from data_loader import My_dataset
from torch.utils.data import DataLoader
from evaluate import *
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
    DuConv_test_DataSet=My_dataset("test",config.data_dir,config.voc_and_embedding_save_path)
    #test时batchsize=1
    test_loader = DataLoader(dataset=DuConv_test_DataSet,\
            shuffle=True, batch_size=1,drop_last=True,collate_fn=collate_fn)
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
        for batch_idx, data in enumerate(test_loader):
            history,knowledge,responses=data["history"],data["knowledge"],data["response"]
            history,len_history=padding_sort_transform(history)
            knowledge,len_knowledge=padding_sort_transform(knowledge)
            responses,len_responses=padding_sort_transform(responses)
            if config.use_gpu and USE_CUDA: 
                history,knowledge,responses = history.cuda() ,\
                    knowledge.cuda() ,responses.cuda()
            encoder_outputs, encoder_hidden = encoder(history,len_history,knowledge,len_knowledge)
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            output_words_list=beam_decode(decoder,decoder_hidden,encoder_outputs,voc,config.beam_size,config.max_dec_len)
            output_words, score=output_words_list[0]
            output_sentence = ' '.join(output_words)
            output_sentences.append(output_sentence)
    topic_materialization(output_sentences,config.data_dir,config.output_path)
    text_path=os.path.join(config.data_dir,"text."+"test"+".txt")
    eval_path=os.path.join(config.data_dir,"result_eval.txt")
    eval(config.output_path,text_path,eval_path)
