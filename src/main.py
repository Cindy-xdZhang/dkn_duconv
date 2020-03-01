# -*- encoding: utf-8 -*-
#'''
#@file_name    :main.py
#@description    :
#@time    :2020/02/12 13:46:28
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import argparse
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_loader import My_dataset
import time
from utils import *
import network 
from optimiser import ScheduledOptim
USE_CUDA = torch.cuda.is_available() 

def arg_config():
    def print_config_information(config):
        print('======================model===============================')
        print('--model_type: '+str(config.model_type))
        print('--dropout: '+str(config.dropout))
        print('--n_heads: '+str(config.n_heads))
        print('--hidden_size: '+str(config.hidden_size))
        print('--n_heads: '+str(config.n_heads))
        print('--n_layers: '+str(config.n_layers))
        print('--attn: '+str(config.attn))
        if config.continue_training==" ":
            print('--continue_training(load model from checkpoint): NONE')
        else :
            print('--continue_training(load model from checkpoint): '+str(config.model_type))
        print('================hyper parameters===========================')
        print('--run_type: '+str(config.run_type))
        print('--batch_size: '+str(config.batch_size))
        print('--optimizer: '+str(config.optimizer))
        print('--learning rate: '+str(config.lr))
        print('--save_iteration: '+str(config.save_iteration))
        print('===========================================================')
    """ config """
    parser = argparse.ArgumentParser()
    # Network CMD参数组
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--model_type", type=str, default='gru',
                         choices=['trans', 'gru'])
    net_arg.add_argument("--hidden_size", type=int, default=10)
    net_arg.add_argument("--n_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='general',
                         choices=['none', 'concat', 'dot', 'general'])
    net_arg.add_argument("--dropout", type=float, default=0)
    net_arg.add_argument("--k_dims", type=int, default=64)
    net_arg.add_argument("--v_dims", type=int, default=64)
    net_arg.add_argument("--n_heads", type=int, default=8)

    # Training / Testing CMD参数组
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--n_warmup_steps", type=int, default=4000)
    train_arg.add_argument("--batch_size", type=int, default=1)
    train_arg.add_argument('-r',"--run_type", type=str, default="train",
     choices=['train', 'test'])
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.002)#for transformer init lr will expand 1000times.so recommendation is 0.002
    train_arg.add_argument("--end_epoch", type=int, default=13)
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--beam_size", type=int, default=3)
    gen_arg.add_argument("--max_dec_len", type=int, default=25,\
        help="limit the length of the decoder output sentense.")
    # MISC ：logs,dirs and gpu config
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument('-u', "--use_gpu", type=str2bool, default=True)
    misc_arg.add_argument('-p',"--log_steps", type=int, default=1)
    misc_arg.add_argument("--save_iteration", type=int, default=20,help='Every save_iteration iteration(s) save checkpoint model ')   
    #路径参数
    misc_arg.add_argument('-i',"--data_dir", type=str,  default="C:\\Users\\10718\\PycharmProjects\\dkn_duconv\\duconv_data",\
        help="The input text data path.")
    misc_arg.add_argument("--voc_and_embedding_save_path", type=str,  default="dkn_duconv",help="The path for voc and embedding file.")
    misc_arg.add_argument("--output_path", type=str, default="dkn_duconv/output/")
    misc_arg.add_argument("--best_model_path", type=str, default="dkn_duconv/models/best_model/")
    misc_arg.add_argument("--save_model_path", type=str, default="dkn_duconv/models")
    misc_arg.add_argument("--continue_training", type=str, default="C:\\Users\\10718\\PycharmProjects\\dkn_duconv\\models\\gru\\L1_H10_general\\Epo_01_iter_00020.tar")
    misc_arg.add_argument("--logfile_path", type=str, default="./log.txt")
    config = parser.parse_args()
    print_config_information(config)
    return config
def build_models(voc,config,checkpoint):
    voc_size=voc.n_words
    hidden_size=config.hidden_size
    if config.model_type =="gru":
        #embedding在encoder 和decoder外面因为他们共用embedding
        embedding_layer = nn.Embedding(voc_size, WORD_EMBEDDING_DIM)
        embedding_layer.weight.data.copy_(torch.from_numpy(build_embedding(voc,config.voc_and_embedding_save_path)))
        encoder = network.EncoderRNN(hidden_size, WORD_EMBEDDING_DIM, embedding_layer, config.n_layers, config.dropout)
        attn_model = config.attn
        decoder = network.LuongAttnDecoderRNN(attn_model, embedding_layer,WORD_EMBEDDING_DIM, hidden_size, voc_size,\
            config.n_layers, config.dropout)
    elif config.model_type =="trans":
        voc_embedding_layer = nn.Embedding(voc_size, WORD_EMBEDDING_DIM)
        voc_embedding_layer.weight.data.copy_(torch.from_numpy(build_embedding(voc,config.voc_and_embedding_save_path)))
        pos_embedding_layer=network.PositionalEncoding(WORD_EMBEDDING_DIM,n_position=300)
        encoder =network.TransfomerEncoder(config,voc_embedding_layer,pos_embedding_layer)
        decoder =network.TransfomerDecoder(config,voc_embedding_layer,pos_embedding_layer,voc_size)
    else: raise Exception("model type error!")

    if checkpoint != None:
        if checkpoint['type'] !=config.model_type:
            raise Exception("checkpoint and train model type doesn't match!")
        print('-loading models from checkpoint .....')
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    if config.use_gpu and USE_CUDA:
        network.Global_device = torch.device("cuda:0" )
        print('**work with GPU **')
    else:
        network.Global_device = torch.device("cpu")
        print('**work with CPU **')
    encoder = encoder.to(network.Global_device)
    decoder = decoder.to(network.Global_device)

    if config.run_type!="train":
        encoder.eval()
        decoder.eval()
    return encoder,decoder

def save_checkpoint(handeler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=handeler
    if config.model_type =="gru":
        save_directory = os.path.join(config.save_model_path,config.model_type,'L{}_H{}_'.format(config.n_layers,config.hidden_size)+config.attn)
        if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        save_path= os.path.join(save_directory,'Epo_{:0>2d}_iter_{:0>5d}.tar'.format(epoch,start_iteration))
        torch.save({
                'epoch': epoch,
                'iteration': start_iteration,
                'type':str(config.model_type),
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, save_path)
    elif config.model_type =="trans":
        save_directory = os.path.join(config.save_model_path,config.model_type,'L{}_H{}_'.format(config.n_layers,config.hidden_size))
        if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        save_path= os.path.join(save_directory,'Epo_{:0>2d}_iter_{:0>5d}.tar'.format(epoch,start_iteration))
        torch.save({
                'epoch': epoch,
                'iteration': start_iteration,
                'type':str(config.model_type),
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, save_path)
    
def train_trans(config):
    if config.batch_size < 2048 and config.n_warmup_steps <= 4000 :
        print('[Warning] The warmup steps may be not enough.\n'\
              '(batch_size, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')
    print("-Loading dataset ...")
    DuConv_DataSet=My_dataset(config.run_type,config.data_dir,config.voc_and_embedding_save_path)
    train_loader = DataLoader(dataset=DuConv_DataSet,\
         shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print('-Building models ...')
    checkpoint =torch.load(config.continue_training,map_location=network.Global_device)  if config.continue_training != " " else None
    encoder,decoder=build_models(DuConv_DataSet.voc,config,checkpoint)
    print('-Building optimizers ...')
    encoder_optimizer = ScheduledOptim(
        optim.Adam(encoder.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.lr*1000, config.hidden_size, config.n_warmup_steps)
    decoder_optimizer = ScheduledOptim(
        optim.Adam(decoder.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.lr*1000, config.hidden_size, config.n_warmup_steps)    

    if checkpoint != None:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
    print('-Initializing training process...')
    start_epoch=1
    start_iteration = 1
    if checkpoint != None:
        start_iteration = checkpoint['iteration'] +1
        start_epoch= checkpoint['epoch'] 
        if start_iteration==int(len(train_loader)//config.batch_size)+1:start_epoch+=1
    end_epoch=config.end_epoch
    
    for epoch_id in range(start_epoch, end_epoch):
        train_handler=(epoch_id,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
        start_iteration+= trainIter_trans(train_handler)        
def trainIter_trans(train_handler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=train_handler
    stage_total_loss=0
    batch_size=config.batch_size
    for batch_idx, data in enumerate(train_loader):
        history,knowledge,responses=data["history"],data["knowledge"],data["response"]
        history = pad_sequence(history,batch_first=True, padding_value=0)
        knowledge = pad_sequence(knowledge,batch_first=True, padding_value=0)
        
        len_respons=[len(it) for it in responses]
        len_respons.sort()
        MAX_RESPONSE_LENGTH=len_respons[-1]-1
        responses =pad_sequence(responses,batch_first=True, padding_value=0)
        responses=responses.transpose(0,1)
        if config.use_gpu and USE_CUDA: 
            history,knowledge,responses= history.cuda() ,knowledge.cuda() ,responses.cuda()
        #清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        #enc_outs =[batchsize,seq,embedding]
        enc_output = encoder(history,knowledge)
        #transformer decoder input 是之前生成的所有句子-》注意观察一下POSITION ENCODING 的移位情况
        decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(batch_size,1) #[batch_size,1]
        decoder_input = decoder_input.to(network.Global_device)
        loss=0
        for t in range(MAX_RESPONSE_LENGTH):
            decoder_output, self_attentions, context_attentions = decoder( decoder_input, enc_output )
            #TODO:我用的选最后一个加squeeze. [b,l,dim]原文中又是如何变成[b,dim]的？
            decoder_output=decoder_output[:,-1,:].squeeze(1)
            #topi为概率最大词汇的下标shape=[batch_Size,1]
            _, topi = decoder_output.topk(1) # [batch_Size, 1]
            decoder_input= torch.LongTensor([decoder_input[i].numpy().tolist()+topi[i].numpy().tolist()    \
                for i in range(batch_size)]).reshape(batch_size,-1)
            decoder_input = decoder_input.to(network.Global_device)  
            # decoder_output=[batch_Size, voc]  responses[seq,batchsize]
            
            loss += F.cross_entropy(decoder_output, responses[t+1], ignore_index=PAD_token)
        loss.backward()
        clip = 50.0
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step_and_update_lr()    
        decoder_optimizer.step_and_update_lr()  
        stage_total_loss+=loss.cpu().item() 
        #相当于更新权重值
        if batch_idx % config.log_steps == 0:
            print_loss_avg = (stage_total_loss / config.log_steps)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
               epoch, batch_idx , len(train_loader),
               100. * batch_idx / len(train_loader), print_loss_avg, time.asctime(time.localtime(time.time())) ))
            with open(config.logfile_path,'a') as f:
                template=' Train Epoch: {} [{}/{}]\tLoss: {:.6f}\ttime: {}\n'
                str=template.format(epoch,batch_idx , len(train_loader),print_loss_avg,\
                    time.asctime(time.localtime(time.time())))
                f.write(str)
            stage_total_loss=0
        if start_iteration % config.save_iteration == 0:
            save_handler=(epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
            save_checkpoint(save_handler)
        start_iteration+=1
    return len(train_loader)
def train_gru(config):
    print('-Loading dataset ...')
    DuConv_DataSet=My_dataset(config.run_type,config.data_dir,config.voc_and_embedding_save_path)
    train_loader = DataLoader(dataset=DuConv_DataSet,\
         shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print('-Building models ...')
    checkpoint =torch.load(config.continue_training,map_location=network.Global_device)  if config.continue_training != " " else None
    encoder,decoder=build_models(DuConv_DataSet.voc,config,checkpoint)
    print('-Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr )
    if checkpoint != None:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
    print('-Initializing training process...')
    start_epoch=1
    start_iteration = 1
    if checkpoint != None:
        start_iteration = checkpoint['iteration'] +1
        start_epoch= checkpoint['epoch'] 
        if start_iteration==int(len(train_loader)//config.batch_size)+1:start_epoch+=1
    end_epoch=config.end_epoch
    
    for epoch_id in range(start_epoch, end_epoch):
        train_handler=(epoch_id,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
        start_iteration+= trainIter_gru(train_handler)   
def trainIter_gru(train_handler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=train_handler
    stage_total_loss=0
    batch_size=config.batch_size
    for batch_idx, data in enumerate(train_loader):
        history,knowledge,responses=data["history"],data["knowledge"],data["response"]
        #log2020.2.23:之前没有发现padding_sort_transform后每个batch内的顺序变了,必须把idx_unsort 也加进来
        history,len_history,idx_unsort1 = padding_sort_transform(history)
        knowledge,len_knowledge,idx_unsort2 = padding_sort_transform(knowledge)
        responses,len_response,idx_unsort3 = padding_sort_transform(responses)
        if config.use_gpu and USE_CUDA: 
            history,knowledge,responses,idx_unsort1,idx_unsort2,idx_unsort3 = history.cuda() ,\
                knowledge.cuda() ,responses.cuda(),idx_unsort1.cuda(),idx_unsort2.cuda(),idx_unsort3.cuda()
        #清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        #encoder_outputs=torch.Size([ 154(seq),2 (batchsize), 512(hiddensize)])
        # encoder_hidden=[ (direction*layer),batchsie,hiddensize]
        unsort_idxs=(idx_unsort1,idx_unsort2)
        encoder_outputs, encoder_hidden = encoder(history,len_history,knowledge,len_knowledge,unsort_idxs)

        decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(1,batch_size) #[batch_size,1]
        decoder_input = decoder_input.to(network.Global_device)
        #decoder 不用双向
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        loss=0
        MAX_RESPONSE_LENGTH=int(len_response[0].item())-1
        responses=responses.index_select(1,idx_unsort3)
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
        loss.backward()
        clip = 50.0
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step()    
        decoder_optimizer.step()  
        stage_total_loss+=loss.cpu().item() 
        #相当于更新权重值
        if batch_idx % config.log_steps == 0:
            print_loss_avg = (stage_total_loss / config.log_steps)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
               epoch, batch_idx , len(train_loader),
               100. * batch_idx / len(train_loader), print_loss_avg, time.asctime(time.localtime(time.time())) ))
            with open(config.logfile_path,'a') as f:
                template=' Train Epoch: {} [{}/{}]\tLoss: {:.6f}\ttime: {}\n'
                str=template.format(epoch,batch_idx , len(train_loader),print_loss_avg,\
                    time.asctime(time.localtime(time.time())))
                f.write(str)
            stage_total_loss=0
        if start_iteration % config.save_iteration == 0:
            save_handler=(epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
            save_checkpoint(save_handler)
        start_iteration+=1
    return len(train_loader)

if __name__ == "__main__":
    #配置解析CMD 参数
    config = arg_config()
    # 模式： train / test
    run_type = config.run_type
    if run_type == "train":
        if config.model_type=="trans":
            train_trans(config)
        elif config.model_type=="gru":
            train_gru(config)
    elif run_type == "test":
        from test import test_model
        test_model(config)
    else: raise ValueError("run_type has to train or test.")
