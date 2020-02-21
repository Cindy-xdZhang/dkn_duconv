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
USE_CUDA = torch.cuda.is_available() 

def arg_config():
    """ config """
    parser = argparse.ArgumentParser()
    # Network CMD参数组
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--hidden_size", type=int, default=128)
    net_arg.add_argument("--n_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='general',
                         choices=['none', 'concat', 'dot', 'general'])
    net_arg.add_argument("--dropout", type=float, default=0.3)
    # Training / Testing CMD参数组
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--batch_size", type=int, default=2)
    train_arg.add_argument('-r',"--run_type", type=str, default="train")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--end_epoch", type=int, default=13)
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--beam_size", type=int, default=3)
    gen_arg.add_argument("--max_dec_len", type=int, default=25,\
        help="limit the length of the decoder output sentense.")
    # MISC ：logs,dirs and gpu config
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument('-u', "--use_gpu", type=str2bool, default=False)
    misc_arg.add_argument('-p',"--log_steps", type=int, default=1)
    misc_arg.add_argument("--save_iteration", type=int, default=5,help='Every save_iteration iteration(s) save checkpoint model ')   
    #路径参数
    misc_arg.add_argument('-i',"--data_dir", type=str,  default="C:\\Users\\10718\\PycharmProjects\\dkn_duconv\\duconv_data",\
        help="The input text data path.")
    misc_arg.add_argument("--voc_and_embedding_save_path", type=str,  default="dkn_duconv",help="The path for voc and embedding file.")
    misc_arg.add_argument("--output_path", type=str, default="./output/test.result")
    misc_arg.add_argument("--best_model_path", type=str, default="dkn_duconv/models/best_model/")
    misc_arg.add_argument("--save_model_path", type=str, default="dkn_duconv/models")
    misc_arg.add_argument("--continue_training", type=str, default=" ")

    config = parser.parse_args()

    return config
def build_models(voc,config,checkpoint):
    voc_size=voc.n_words
    hidden_size=config.hidden_size
    #embedding在encoder 和decoder外面因为他们共用embedding
    embedding_layer = nn.Embedding(voc_size, WORD_EMBEDDING_DIM)
    embedding_layer.weight.data.copy_(torch.from_numpy(build_embedding(voc,config.voc_and_embedding_save_path)))
    encoder = network.EncoderRNN(hidden_size, WORD_EMBEDDING_DIM, embedding_layer, config.n_layers, config.dropout)
    attn_model = config.attn

    decoder = network.LuongAttnDecoderRNN(attn_model, embedding_layer,WORD_EMBEDDING_DIM, hidden_size, voc_size,\
        config.n_layers, config.dropout)
    if checkpoint != None:
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    if config.use_gpu and USE_CUDA:
        network.Global_device = torch.device("cuda:0" )
        print('**Train with GPU **')
    else:
        network.Global_device = torch.device("cpu")
        print('**Train with CPU **')
    encoder = encoder.to(network.Global_device)
    decoder = decoder.to(network.Global_device)
    embedding_layer=embedding_layer.to(network.Global_device)
    if config.run_type!="train":
        encoder.eval()
        decoder.eval()
        embedding_layer.eval()
    return encoder,decoder

def save_checkpoint(handeler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=handeler
    save_directory = os.path.join(config.save_model_path, 'L{}_H{}_'.format(config.n_layers,config.hidden_size)+config.attn)
    if not os.path.exists(save_directory):
                os.makedirs(save_directory)
    save_path= os.path.join(save_directory,'Epo_{:0>2d}_iter_{:0>5d}.tar'.format(epoch,start_iteration))
    torch.save({
                'epoch': epoch,
                'iteration': start_iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, save_path)
def train(config):
    print('-Loading dataset...')
    DuConv_DataSet=My_dataset(config.run_type,config.data_dir,config.voc_and_embedding_save_path)
    train_loader = DataLoader(dataset=DuConv_DataSet,\
         shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print('-Building models...')
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
        start_iteration = checkpoint['iteration'] + 1
        start_epoch= checkpoint['epoch'] + 1
    end_epoch=config.end_epoch
    
    for epoch_id in range(start_epoch, end_epoch):
        train_handler=(epoch_id,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
        print('-training epoch '+str(epoch_id)+" ...")
        start_iteration+= trainIter(train_handler)
        

def trainIter(train_handler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=train_handler
    stage_total_loss=0
    batch_size=config.batch_size
    for batch_idx, data in enumerate(train_loader):
        history,knowledge,responses=data["history"],data["knowledge"],data["response"]
        history,len_history=padding_sort_transform(history)
        knowledge,len_knowledge=padding_sort_transform(knowledge)
        responses,len_responses=padding_sort_transform(responses)
        if config.use_gpu and USE_CUDA: 
            history,knowledge,responses = history.cuda() ,\
                knowledge.cuda() ,responses.cuda()
        #清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        #encoder_outputs=torch.Size([ 154(seq),2 (batchsize), 512(hiddensize)])
        # encoder_hidden=[ (direction*layer),batchsie,hiddensize]
        encoder_outputs, encoder_hidden = encoder(history,len_history,knowledge,len_knowledge)

        decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(1,batch_size) #[batch_size,1]
        decoder_input = decoder_input.to(network.Global_device)
        #decoder 不用双向
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        loss=0
        MAX_RESPONSE_LENGTH=int(len_responses[0].item())-1
        for t in range(MAX_RESPONSE_LENGTH):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            #topi为概率最大词汇的下标
            _, topi = decoder_output.topk(1) # [batch_Size, 1]

            decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).reshape(1,batch_size)
            decoder_input = decoder_input.to(network.Global_device)    
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
            with open('dkn_duconv/log.txt','a') as f:
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
        train(config)
    elif run_type == "test":
        from test import test_model
        test_model(config)
    else: raise ValueError("run_type has to train or test.")
