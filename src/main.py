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
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
USE_CUDA = torch.cuda.is_available()

def arg_config():
    """ config """
    parser = argparse.ArgumentParser()

    # Data CMD参数组
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    # data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--vocab_path", type=str, default="./data/vocab.txt")
    data_arg.add_argument("--embed_file", type=str,
                          default="./data/sgns.weibo.300d.txt")

    # Network CMD参数组
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    # 训练时由载入的vocab又重新更新了以下vocab_size
    net_arg.add_argument("--vocab_size", type=int, default=30004)
    #过滤知识三元组时的filter参数 单个实体名长度大于等于min_len 小于等于max_len
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])

    # Training / Testing CMD参数组
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument('-r',"--run_type", type=str, default="train")
    train_arg.add_argument("--continue", type=str, default="")
    train_arg.add_argument("--init_model", type=str, default="")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--end_epochs", type=int, default=13)
    train_arg.add_argument("--start_epochs", type=int, default=1)
    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--beam_size", type=int, default=10)
    gen_arg.add_argument("--max_dec_len", type=int, default=30)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--output_path", type=str, default="./output/test.result")
    gen_arg.add_argument("--best_model_path", type=str, default="./models/best_model/")
    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument('-u', "--use_gpu", type=str2bool, default=True)
    misc_arg.add_argument('-p',"--log_steps", type=int, default=300)
    misc_arg.add_argument("--valid_steps", type=int, default=1000)
    misc_arg.add_argument("--batch_size", type=int, default=1)

    config = parser.parse_args()

    return config
def train(config):
    # load_Data()
    # train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    # test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size, **kwargs)
    device = torch.device("cuda:0" if USE_CUDA and config.use_gpu else "cpu")
    if config.use_gpu:
        model = model.cuda()
        print('**Train using GPU **')
    else:
        print('**Train usingSE CPU **')
    #model = LeNet5()
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    for epoch_id in range(start_epochs, end_epochs):
        trainIter()

   
    
    
def trainIter():
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)                      #定义为Variable类型，能够调用autograd
        #初始化时，要清空梯度
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()                                                     #相当于更新权重值
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == "__main__":
    #配置解析CMD 参数
    config = arg_config()
    # 模式： train / test
    run_type = config.run_type
    
    if run_type == "train":
        train(config)
    elif run_type == "test":
        test(config)
