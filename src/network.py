# -*- encoding: utf-8 -*-
#'''
#@file_name    :network.py
#@description    :
#@time    :2020/02/13 12:47:27
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer_sublayers import *
from torch.nn.utils.rnn import pad_sequence
from utils import *
Global_device="cpu"
MAX_LENGTH=500
WORD_EMBEDDING_DIM_NO_PRETRAIN=25
def save_checkpoint(handeler):
    epoch,start_iteration,emb,encoder,decoder,optimizer,config=handeler
    if config.model_type =="gru":
        save_directory = os.path.join(config.save_model_path,config.model_type,'L{}_H{}_'.format(config.n_layers,config.hidden_size)+config.attn)
        if not os.path.exists(save_directory):
                os.makedirs(save_directory)
        save_path= os.path.join(save_directory,'Epo_{:0>2d}_iter_{:0>6d}.tar'.format(epoch,start_iteration))
        torch.save({
                'epoch': epoch,
                'iteration': start_iteration,
                'type':str(config.model_type),
                'emb':emb.state_dict(),
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'opt': optimizer.state_dict(),
            }, save_path)
    elif config.model_type =="trans":
        pass
        # save_directory = os.path.join(config.save_model_path,config.model_type,'L{}_H{}'.format(config.n_layers,config.hidden_size))
        # if not os.path.exists(save_directory):
        #         os.makedirs(save_directory)
        # save_path= os.path.join(save_directory,'Epo_{:0>2d}_iter_{:0>6d}.tar'.format(epoch,start_iteration))
        # torch.save({
        #         'epoch': epoch,
        #         'iteration': start_iteration,
        #         'type':str(config.model_type),
        #         'en': encoder.state_dict(),
        #         'de': decoder.state_dict(),
        #         'en_opt': encoder_optimizer.state_dict(),
        #         'de_opt': decoder_optimizer.state_dict(),
        #     }, save_path)    
#=========================GRU seq2seq=================================
class EncoderRNN_noKG(nn.Module):
    def __init__(self, hidden_size, embedding_size, n_layers=1, dropout=0):
        super(EncoderRNN_noKG, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru_History = nn.GRU(embedding_size, hidden_size, n_layers,
                        dropout=(0 if n_layers == 1 else dropout), bidirectional=True,batch_first =False)
        torch.nn.init.orthogonal_( self.gru_History.weight_hh_l0)
        torch.nn.init.orthogonal_( self.gru_History.weight_hh_l0_reverse)
        torch.nn.init.orthogonal_( self.gru_History.weight_ih_l0)
        torch.nn.init.orthogonal_( self.gru_History.weight_ih_l0_reverse)
        # hidden_kg=( num_layers * num_directions, batch,hidden_size)
        # batch first 只影响output 不影响hidden的形状 所以batch first=false格式更统一
        # self.W1=torch.nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        # self.PReLU1=torch.nn.PReLU()

    def forward(self, input_history_seq_embedded,input_history_lengths,input_kg_seq,input_kg_lengths, unsort_idxs):
        unsort_idx_history,unsort_idx_kg=unsort_idxs
        #history
        # input_history_seq_embedded = self.embedding(input_history_seq)
        input_history_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_history_seq_embedded, input_history_lengths,batch_first=False)
        his_outputs, hidden_his= self.gru_History(input_history_seq_packed, None)
        his_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(his_outputs,batch_first=False)
        his_outputs=his_outputs.index_select(1,unsort_idx_history)
        hidden_his=hidden_his.index_select(1,unsort_idx_history)
        his_outputs = his_outputs[:, :, :self.hidden_size] + his_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (batch, 1, hidden)
        return his_outputs, hidden_his
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru_KG = nn.GRU(embedding_size, hidden_size, n_layers,
                        dropout=(0 if n_layers == 1 else dropout), bidirectional=True,batch_first =False)
        self.gru_History = nn.GRU(embedding_size, hidden_size, n_layers,
                        dropout=(0 if n_layers == 1 else dropout), bidirectional=True,batch_first =False)
        torch.nn.init.orthogonal_( self.gru_KG.weight_hh_l0)
        torch.nn.init.orthogonal_( self.gru_KG.weight_hh_l0_reverse)
        torch.nn.init.orthogonal_( self.gru_KG.weight_ih_l0)
        torch.nn.init.orthogonal_( self.gru_KG.weight_ih_l0_reverse)
        torch.nn.init.orthogonal_( self.gru_History.weight_hh_l0)
        torch.nn.init.orthogonal_( self.gru_History.weight_hh_l0_reverse)
        torch.nn.init.orthogonal_( self.gru_History.weight_ih_l0)
        torch.nn.init.orthogonal_( self.gru_History.weight_ih_l0_reverse)
        #GRU的 output: (seq_len, batch, hidden*n_dir) ,
        # hidden_kg=( num_layers * num_directions, batch,hidden_size)
        # batch first 只影响output 不影响hidden的形状 所以batch first=false格式更统一
        self.W1=torch.nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.W2=torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.PReLU1=torch.nn.PReLU()
        # self.PReLU2=torch.nn.PReLU()

    def forward(self, input_history_seq,input_history_lengths,input_kg_seq,input_kg_lengths, unsort_idxs):
        unsort_idx_history,unsort_idx_kg=unsort_idxs
        #kg
        #input_kg_seq_embedded [seq,batchsize, embeddingsize]
        input_kg_seq_embedded = self.embedding(input_kg_seq)
        #【seq*batch*embed_dim】
        input_kg_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_kg_seq_embedded, input_kg_lengths,batch_first=False)
        #GRU的 output: (seq_len, batch, hidden*n_dir) ,
        # hidden_kg=( num_layers * num_directions, batch,hidden_size)
        kg_outputs,hidden_kg=self.gru_KG(input_kg_seq_packed, None) 
        kg_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(kg_outputs,batch_first=False )
        kg_outputs=kg_outputs.index_select(1,unsort_idx_kg)
        hidden_kg=hidden_kg.index_select(1,unsort_idx_kg)
        kg_outputs = kg_outputs[:, :, :self.hidden_size] + kg_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs ( batch,1, hidden)
        
        #history
        input_history_seq_embedded = self.embedding(input_history_seq)
        input_history_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_history_seq_embedded, input_history_lengths,batch_first=False)
        his_outputs, hidden_his= self.gru_History(input_history_seq_packed, None)
        his_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(his_outputs,batch_first=False)
        his_outputs=his_outputs.index_select(1,unsort_idx_history)
        hidden_his=hidden_his.index_select(1,unsort_idx_history)
        his_outputs = his_outputs[:, :, :self.hidden_size] + his_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (batch, 1, hidden)
        # hidden_kg=(num_layers * num_directions,batch,  hidden_size)
        concat_hidden=torch.cat((hidden_his, hidden_kg),1).reshape(self.n_layers*2,-1, self.hidden_size*2)
        hidden=self.W1(concat_hidden)
        outputs=self.W2(torch.cat((his_outputs, kg_outputs), 0))
        return outputs, hidden
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden [1, 64, 512], encoder_outputs [14, 64, 512]
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, max_len) # B x S
        attn_energies = attn_energies.to(Global_device)

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        # hidden [1, 512], encoder_output [1, 512]
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model,embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,\
            dropout=(0 if n_layers == 1 else dropout), batch_first=False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq_embedded, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # embedded = self.embedding_dropout(embedded) 

        # Get current hidden state from input word and last hidden state
        #batch_first=True 不影响hidden初始化的格式是(num_layers * num_directions, batch, hidden_size)
        rnn_output, hidden = self.gru(input_seq_embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #[batchsize, 1, 14]
        # encoder_outputs [seq, batchsize, hiddensize]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[batchsize, 1, hiddensize]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) #[batchsize, hiddensize]
        context = context.squeeze(1) #[batchsize4, hiddensize]
        concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
        concat_output = torch.tanh(self.concat(concat_input)) #[64, 512]

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #[batchsize, output_size(vocabularysize)]

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
class GRU_Encoder_Decoder(nn.Module):
    def __init__(self, config,voc):
        super(GRU_Encoder_Decoder, self).__init__()
        voc_size=voc.n_words
        print('-Building GRU_Encoder_Decoder ...')
        self.config=config
        WORD_EMBEDDING_DIMs=WORD_EMBEDDING_DIM_PRETRAIN if config.pre_train_embedding==True \
            else WORD_EMBEDDING_DIM_NO_PRETRAIN
        self.embedding=nn.Embedding(voc_size, WORD_EMBEDDING_DIMs,padding_idx=PAD_token)
        if config.pre_train_embedding==True:self.embedding.weight.data.copy_(torch.from_numpy(\
            build_embedding(voc,config.voc_and_embedding_save_path)))
        self.encoder=EncoderRNN_noKG(config.hidden_size,WORD_EMBEDDING_DIMs,config.n_layers,config.dropout)
        self.decoder=LuongAttnDecoderRNN(config.attn,WORD_EMBEDDING_DIMs, config.hidden_size, voc_size, n_layers=config.n_layers, dropout=config.dropout)
        encoder_para = sum([np.prod(list(p.size())) for p in self.encoder.parameters()])
        decoder_para = sum([np.prod(list(p.size())) for p in self.decoder.parameters()])
        print('Build encoder with params: {:4f}M'.format( encoder_para * 4 / 1000 / 1000))
        print('Build decoder with params: {:4f}M'.format( decoder_para * 4 / 1000 / 1000))
        #loading
        checkpoint =torch.load(config.continue_training,map_location=Global_device) \
            if config.continue_training != " " else None
        if checkpoint != None:
            if checkpoint['type'] !=config.model_type:
                raise Exception("checkpoint and train model type doesn't match!")
            print('-loading models from checkpoint .....')
            self.encoder.load_state_dict(checkpoint['en'])
            self.decoder.load_state_dict(checkpoint['de'])
            self.embedding.load_state_dict(checkpoint['emb'])
        self.checkpoint=checkpoint
        self.teacher_forcing_ratio=1
    def forward(self, BatchData):
        batch_size=self.config.batch_size
        history,knowledge,responses=BatchData["history"],BatchData["knowledge"],BatchData["response"]
        #log2020.2.23:之前没有发现padding_sort_transform后每个batch内的顺序变了,必须把idx_unsort 也加进来
        history,len_history,idx_unsort1 = padding_sort_transform(history)
        knowledge,len_knowledge,idx_unsort2 = padding_sort_transform(knowledge)
        responses,len_response,idx_unsort3 = padding_sort_transform(responses)
        history,idx_unsort1 =history.to(Global_device),idx_unsort1.to(Global_device) 
        knowledge,idx_unsort2= knowledge.to(Global_device),idx_unsort2.to(Global_device)
        responses,idx_unsort3 = responses.to(Global_device),idx_unsort3.to(Global_device)
        #encoder_outputs=torch.Size([ 154(seq),2 (batchsize), 512(hiddensize)])
        # encoder_hidden=[ (direction*layer),batchsie,hiddensize]
        unsort_idxs=(idx_unsort1,idx_unsort2)
        history= self.embedding(history)
        encoder_outputs, encoder_hidden = self.encoder(history,len_history,knowledge,len_knowledge,unsort_idxs)

        decoder_input = torch.LongTensor([SOS_token for _ in range(batch_size)]).reshape(1,batch_size) #[batch_size,1]
        decoder_input = decoder_input.to(Global_device)
        #decoder 不用双向
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        loss=0
        MAX_RESPONSE_LENGTH=int(len_response[0].item())-1
        responses=responses.index_select(1,idx_unsort3)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing == False :
            for t in range(MAX_RESPONSE_LENGTH):
                decoder_input= self.embedding(decoder_input)
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                #topi为概率最大词汇的下标
                _, topi = decoder_output.topk(1) # [batch_Size, 1]

                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).reshape(1,batch_size)
                decoder_input = decoder_input.to(Global_device)  
                # decoder_output=[batch_Size, voc]  responses[seq,batchsize]
                loss += F.cross_entropy(decoder_output, responses[t+1], ignore_index=EOS_token)
        else:
            for t in range(MAX_RESPONSE_LENGTH):
                decoder_input= self.embedding(decoder_input)
                decoder_output, decoder_hidden, decoder_attn = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_input = responses[t+1].view(1, -1) # Next input is current target
                decoder_input = decoder_input.to(Global_device)  
                loss += F.cross_entropy(decoder_output, responses[t+1], ignore_index=EOS_token)
        return loss,MAX_RESPONSE_LENGTH        
    def train(self, train_loader,dev_loader,optimizer):
        print('-Initializing training process...')
        start_epoch=1
        start_iteration = 1
        if self.checkpoint != None:
            start_iteration = self.checkpoint['iteration'] +1
            start_epoch= self.checkpoint['epoch'] 
        if start_iteration==int(len(train_loader)//self.config.batch_size)+1:start_epoch+=1
        end_epoch=self.config.end_epoch
        for epoch_id in range(start_epoch, end_epoch):
            iterations,epoch_loss= self.trainIter_gru(epoch_id,start_iteration,train_loader,optimizer)   
            start_iteration+=iterations
            record_train_step(self.config.logfile_path,epoch_id,epoch_loss)
            self.dev(epoch_id,dev_loader)
    def trainIter_gru(self, epoch,start_iteration,train_loader,optimizer):
        self.encoder.train()
        self.decoder.train()
        self.embedding.train()
        stage_total_loss=0
        epoch_loss_avg=0
        batch_size=self.config.batch_size
        self.teacher_forcing_ratio=1/(epoch**(-2))
        for batch_idx, data in enumerate(train_loader):
            batch_idx+=1
            #清空梯度
            optimizer.zero_grad()
            loss,nwords=self.forward(data)
            loss.backward()
            clip = 100.0
            _ = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()    
            stage_total_loss+=(loss.cpu().item()/nwords)
            epoch_loss_avg+=(loss.cpu().item() /nwords)
            #相当于更新权重值
            if batch_idx % self.config.log_steps == 0:
                print_loss_avg = (stage_total_loss / self.config.log_steps)
                message=epoch,batch_idx , len(train_loader),print_loss_avg
                record_train_step(self.config.logfile_path,message)
                stage_total_loss=0
            if start_iteration % self.config.save_iteration == 0:
                save_handler=(epoch,start_iteration, self.embedding,self.encoder,self.decoder,optimizer,self.config)
                save_checkpoint(save_handler)
            start_iteration+=1
        return len(train_loader),epoch_loss_avg/len(train_loader)
    def dev(self, epoch,dev_loader):
        self.encoder.eval()
        self.decoder.eval()
        self.embedding.eval()
        batch_size=self.config.batch_size
        epoch_loss_avg=0
        with torch.no_grad():
            for batch_idx, data in enumerate(dev_loader):
                loss,nwords=self.forward(data)
                epoch_loss_avg+=(loss.cpu().item() /nwords)
        epoch_loss_avg/=len(dev_loader)
        print('Evaluate Epoch: {}\t avg Loss: {:.6f}\ttime: {}'.format(
            epoch,epoch_loss_avg, time.asctime(time.localtime(time.time())) ))
        with open(self.config.logfile_path,'a') as f:
            template=' Evaluate Epoch: {}\t avg Loss: {:.6f}\ttime: {}\n'
            str=template.format(epoch,epoch_loss_avg,\
                time.asctime(time.localtime(time.time())))
            f.write(str)
#====================Transfomer========================================
class TransformerEncoder(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,config,voc_Size,embedding_layer):
        super().__init__()
        self.char_embedding=embedding_layer
        self.Positional_Encoding=PositionalEncoding(d_hid=config.embedding_size, n_position=MAX_LENGTH)
        self.layerstack=nn.ModuleList([ Encoder_layer(config.embedding_size,config.n_head,config.d_k,config.d_v,config.d_hidden,config.dropout) for _ in range(config.n_layers)  ])

    def forward(self, x):
        #X=B,L
        slf_attn_mask=padding_mask(x)
        output=self.Positional_Encoding(self.char_embedding(x))
        for layer in self.layerstack:
            output = layer(output,slf_attn_mask=slf_attn_mask)
        return output
class TransformerDecoder(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self,config,voc_Size,embedding_layer):
        super().__init__()
        self.Positional_Encoding=PositionalEncoding(d_hid=config.embedding_size, n_position=MAX_LENGTH)
        self.layerstack=nn.ModuleList([ Decoder_layer(config.embedding_size,config.n_head,config.d_k,config.d_v,config.d_hidden,config.dropout) for _ in range(config.n_layers)  ])
        self.char_embedding=embedding_layer
    def forward(self, dec_input,enc_output,enc_input):
        #X=B,L
        slf_attn_mask=padding_mask(dec_input).to(Global_device)
        sq_mask=sequence_mask(dec_input).to(Global_device)
        slf_attn_mask = (torch.gt((slf_attn_mask.float() + sq_mask.float()), 0)).float().to(Global_device)
        enc_dec_mask=get_attn_pad_mask(dec_input,enc_input).to(Global_device)

        output=self.Positional_Encoding(self.char_embedding(dec_input))
        

        for layer in self.layerstack:
            output = layer(output,self_attn_mask=slf_attn_mask,enc_out=enc_output,enc_dec_mask=enc_dec_mask)
        return output
class Transformer(nn.Module):
    def __init__(self, config,voc_Size):
        super().__init__() 
        self.config=config
        self.char_embedding= Embeddings(voc_Size,config.embedding_size)
        self.encoder=TransformerEncoder(config,voc_Size,self.char_embedding)
        self.decoder=TransformerDecoder(config,voc_Size,self.char_embedding)
        self.tgt_proj=nn.Linear(config.embedding_size, voc_Size, bias=False)
        self.final_softmax = nn.Softmax(dim=2)
    def call(self,Q,A):
        enc_output=self.encoder(Q)
        dec_input=A[:,:-1]
        dec_target=A[:,1:]
        dec_input= dec_input.to(Global_device)
        dec_target= dec_target.to(Global_device)
        dec_out=self.decoder(dec_input,enc_output,Q)
        dec_logits = self.final_softmax(self.tgt_proj(dec_out)) 
        preds=dec_logits.contiguous().view(dec_logits.size(0)*dec_logits.size(1),-1)
        tars=dec_target.contiguous().view(-1)
        loss= F.cross_entropy(preds,tars)
        return loss
    def train(self,train_loader,optimizer):
        start_epoch=0
        stage_total_loss=0
        for epoch in range(start_epoch,self.config.end_epoch):
            for batch_idx,batch in enumerate(train_loader):
                batch_idx+=1
                optimizer.zero_grad()
                batchQ,batchA=batch["history"],batch["response"]
                batchQ = pad_sequence(batchQ,batch_first=True, padding_value=0).to(Global_device)
                batchA = pad_sequence(batchA,batch_first=True, padding_value=0).to(Global_device)
                loss=self.call(batchQ,batchA)
                loss.backward()
                optimizer.step_and_update_lr()
                stage_total_loss+=loss.cpu().item() 
                if batch_idx % self.config.log_steps == 0:
                    print_loss_avg = (stage_total_loss / self.config.log_steps)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttime: {}'.format(
                        epoch, batch_idx , len(train_loader),
                        100. * batch_idx / len(train_loader), print_loss_avg, time.asctime(time.localtime(time.time())) ))
                    with open(self.config.logfile_path,'a') as f:
                        template=' Train Epoch: {} [{}/{}]\tLoss: {:.6f}\ttime: {}\n'
                        str=template.format(epoch,batch_idx , len(train_loader),print_loss_avg,\
                            time.asctime(time.localtime(time.time())))
                        f.write(str)
                    stage_total_loss=0
