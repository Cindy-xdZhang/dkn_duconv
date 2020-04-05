# -*- encoding: utf-8 -*-
#'''
#@file_name    :main.py
#@description    :
#@time    :2020/02/12 13:46:28
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import os
import time
import network 
import argparse
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from data_loader import My_dataset
from optimiser import ScheduledOptim
from transformer_sublayers import  get_attn_pad_mask
from utils import *
from test import test_model
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
WORD_EMBEDDING_DIM_NO_PRETRAIN=16
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
        print('--select_kg: '+str(config.select_kg))
        print('--use pre_train_embedding: '+str(config.pre_train_embedding))
        print('--shareW: '+str(config.shareW))
        if config.continue_training==" ":
            print('--continue_training(load model from checkpoint): NONE')
        else :
            print('--continue_training(load model from checkpoint): '+str(config.continue_training))
        print('================hyper parameters===========================')
        print('--run_type: '+str(config.run_type))
        print('--batch_size: '+str(config.batch_size))
        print('--learning rate: '+str(config.lr))
        print('--save_iteration: '+str(config.save_iteration))
        print('===========================================================')
    """ config """
    project_root_dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    parser = argparse.ArgumentParser()
    # Network CMD参数组
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("-m","--model_type", type=str, default='gru',
                         choices=['trans', 'gru'])
    net_arg.add_argument('-eb',"--embedding_size", type=int, default=128,help="for transformer")
    net_arg.add_argument('-hi',"--hidden_size", type=int, default=256)
    net_arg.add_argument("--n_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='general',
                         choices=['none', 'concat', 'dot', 'general'])
    net_arg.add_argument('-d',"--dropout", type=float, default=0.1)
    net_arg.add_argument("--k_dims", type=int, default=64)
    net_arg.add_argument("--v_dims", type=int, default=64)
    net_arg.add_argument("--n_heads", type=int, default=8)
    net_arg.add_argument("--shareW", type=str2bool, default=False)
    net_arg.add_argument('-sk', "--select_kg", type=str2bool, default=True)
    net_arg.add_argument('-pre', "--pre_train_embedding", type=str2bool, default=False)

    # Training / Testing CMD参数组
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--n_warmup_steps", type=int, default=4000)
    train_arg.add_argument('-bs',"--batch_size", type=int, default=16)
    train_arg.add_argument('-r',"--run_type", type=str, default="train",
     choices=['train', 'test'])
    train_arg.add_argument('-lr',"--lr", type=float, default=0.001)
    train_arg.add_argument("--end_epoch", type=int, default=80)
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--beam_size", type=int, default=3)
    gen_arg.add_argument("--max_dec_len", type=int, default=25,\
        help="limit the length of the decoder output sentense.")
    # MISC ：logs,dirs and gpu config
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument('-u', "--use_gpu", type=str2bool, default=True)
    misc_arg.add_argument('-p',"--log_steps", type=int, default=10)
    misc_arg.add_argument('-s',"--save_iteration", type=int, default=1000,help='Every save_iteration iteration(s) save checkpoint model ')   
    #路径参数
    misc_arg.add_argument('-i',"--data_dir", type=str,  default=os.path.join(project_root_dir,"duconv_data"),\
        help="The input text data path.")
    misc_arg.add_argument("--voc_and_embedding_save_path", type=str,  default=project_root_dir,help="The path for voc and embedding file.")
    misc_arg.add_argument("--output_path", type=str, default=os.path.join(project_root_dir,"output"))
    misc_arg.add_argument("--save_model_path", type=str, default=os.path.join(project_root_dir,"models"))
    misc_arg.add_argument('-con',"--continue_training", type=str, default=" ")
    misc_arg.add_argument('-log',"--logfile_path", type=str, default=os.path.join(project_root_dir,"log.txt"))
    config = parser.parse_args()
    print_config_information(config)
    if os.path.exists(config.logfile_path): 
        os.remove(config.logfile_path) 
    return config
def train_trans(config):
    if config.batch_size < 2048 and config.n_warmup_steps < 4000 :
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
    encoder.train()
    decoder.train()
    print('-Building optimizers ...')
    encoder_optimizer = ScheduledOptim(
        optim.Adam(encoder.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.lr, config.hidden_size, config.n_warmup_steps)
    decoder_optimizer = ScheduledOptim(
        optim.Adam(decoder.parameters(), betas=(0.9, 0.98), eps=1e-09),
        config.lr, config.hidden_size, config.n_warmup_steps)    

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
    from test import dev
    DuConv_dev_DataSet=My_dataset("dev",config.data_dir,config.voc_and_embedding_save_path)
    dev_loader = DataLoader(dataset=DuConv_dev_DataSet,\
            shuffle=True, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    for epoch_id in range(start_epoch, end_epoch):
        train_handler=(epoch_id,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
        iterations,epoch_loss= trainIter_trans(train_handler)   
        start_iteration+=iterations
        with open(config.logfile_path,'a') as f:
                template=' Train Epoch: {} \t Overall Loss: {:.6f}\t time: {}\n'
                str=template.format(epoch_id, epoch_loss,time.asctime(time.localtime(time.time())))
                print(str)
                f.write(str)
        dev_handeler=(encoder,decoder,config,epoch_id,DuConv_DataSet.voc,dev_loader)
        dev(dev_handeler)
def trainIter_trans(train_handler):
    epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config=train_handler
    stage_total_loss=0
    epoch_loss_avg=0
    batch_size=config.batch_size
    for batch_idx, data in enumerate(train_loader):
        batch_idx+=1
        history,knowledge,responses=data["history"],data["knowledge"],data["response"]
        #[B,L]
        history = pad_sequence(history,batch_first=True, padding_value=0).to(network.Global_device)
        knowledge = pad_sequence(knowledge,batch_first=True, padding_value=0).to(network.Global_device)
        responses =pad_sequence(responses,batch_first=True, padding_value=0).to(network.Global_device)
        decoder_input =responses[:,:-1].to(network.Global_device)
        decoder_target =responses[:,1:].to(network.Global_device)
        #清空梯度
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        #enc_outs =[batchsize,L,embedding]
        enc_output = encoder(history,knowledge)

        dec_enc_attn_pad_mask =get_attn_pad_mask(decoder_input, history).to(network.Global_device)
        #decoder_output=[B*(L-1),VOCSIZE]
        decoder_output= decoder( decoder_input, enc_output ,dec_enc_attn_pad_mask)
        loss = F.cross_entropy(decoder_output, decoder_target.contiguous().view(-1), ignore_index=PAD_token)
        loss.backward()
        clip = 100.0
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        encoder_optimizer.step_and_update_lr()    
        decoder_optimizer.step_and_update_lr()  
        stage_total_loss+=loss.cpu().item() 
        epoch_loss_avg+=loss.cpu().item() 
        #相当于更新权重值
        if batch_idx % config.log_steps == 0:
            print_loss_avg = (stage_total_loss / config.log_steps)
            message=epoch,batch_idx , len(train_loader),print_loss_avg
            record_train_step()
            stage_total_loss=0
        if start_iteration % config.save_iteration == 0:
            save_handler=(epoch,start_iteration,train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,config)
            save_checkpoint(save_handler)
        start_iteration+=1
    return len(train_loader),epoch_loss_avg/len(train_loader)
def main():
    config=arg_config()
    if config.run_type=='train':
        print("-Loading dataset ...")
        DataSet=My_dataset('train')
        train_loader = DataLoader(dataset=DataSet,\
            shuffle=False, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
        Dev_DataSet=My_dataset('dev',voc=DataSet.voc)
        dev_loader = DataLoader(dataset=DataSet,\
            shuffle=False, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
        if config.model_type=='gru':
            model=network.GRU_Encoder_Decoder(config,DataSet.voc)
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-09)
        elif  config.model_type=='trans':
            model=network.Transformer(config,DataSet.voc.n_words)
            optimizer = ScheduledOptim(
                optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                config.lr, config.embedding_size, config.n_warmup_steps)
        if  torch.cuda.is_available()  and config.use_gpu :
            print('**work with solo-GPU **')
            network.Global_device = torch.device("cuda:0" )
            model.to(network.Global_device)
        else:print('**work with CPU **')
        # for n,p in model.named_parameters(): print(n)
        model.train(train_loader,dev_loader,optimizer)
    else:
        test_model(config)
if __name__ == "__main__":
    main()

