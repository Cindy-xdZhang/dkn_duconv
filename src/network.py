# -*- encoding: utf-8 -*-
#'''
#@file_name    :network.py
#@description    :
#@time    :2020/02/13 12:47:27
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer_sublayers import *
Global_device="cpu"
#=========================GRU seq2seq=================================
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru_KG = nn.GRU(embedding_size, hidden_size, n_layers,
                        dropout=(0 if n_layers == 1 else dropout), bidirectional=True,batch_first =False)
        self.gru_History = nn.GRU(embedding_size, hidden_size, n_layers,
                        dropout=(0 if n_layers == 1 else dropout), bidirectional=True,batch_first =False)
        #GRU的 output: (seq_len, batch, hidden*n_dir) ,
        # hidden_kg=( num_layers * num_directions, batch,hidden_size)
        # batch first 只影响output 不影响hidden的形状 所以batch first=false格式更统一
        self.W1=torch.nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.W2=torch.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.PReLU1=torch.nn.PReLU()
        self.PReLU2=torch.nn.PReLU()

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
        kg_outputs = kg_outputs[:, :, :self.hidden_size] + kg_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs ( batch,1, hidden)
        #history
        input_history_seq_embedded = self.embedding(input_history_seq)
        input_history_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_history_seq_embedded, input_history_lengths,batch_first=False)
        his_outputs, hidden_his= self.gru_History(input_history_seq_packed, None)
        his_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(his_outputs,batch_first=False)
        his_outputs=his_outputs.index_select(1,unsort_idx_history)
        his_outputs = his_outputs[:, :, :self.hidden_size] + his_outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (batch, 1, hidden)
        # hidden_kg=(num_layers * num_directions,batch,  hidden_size)
        concat_hidden=torch.cat((hidden_his, hidden_kg),1).reshape(self.n_layers*2,-1, self.hidden_size*2)
        hidden=self.PReLU1(self.W1(concat_hidden))
        outputs=self.PReLU2(self.W2(torch.cat((his_outputs, kg_outputs), 0)))
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
    def __init__(self, attn_model, embedding,embedding_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        # self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,\
            dropout=(0 if n_layers == 1 else dropout), batch_first=False)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        embedded = self.embedding(input_seq)#[batch,1,embedding_size]
        # embedded = self.embedding_dropout(embedded) 
        if(embedded.size(0) != 1):
            raise ValueError('Decoder input sequence length should be 1')

        # Get current hidden state from input word and last hidden state
        #batch_first=True 不影响hidden初始化的格式是(num_layers * num_directions, batch, hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)

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
#====================Transfomer========================================
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid=300, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
class TransfomerEncoder(nn.Module):
    def __init__(self, config, embedding,pos_embedding,embedding_size=300,dropout=0):
        super(TransfomerEncoder, self).__init__()
        # embedding
        self.char_embedding =embedding
        self.pos_embedding = pos_embedding
        self.dropout=  nn.Dropout(dropout)
        if config.shareW==True:
            d_k=config.k_dims//config.n_heads
            d_v=config.k_dims//config.n_heads
            self.mh_w_qs1=nn.Linear(embedding_size, config.n_heads * d_k)
            self.mh_w_vs1=nn.Linear(embedding_size, config.n_heads * d_v)
            self.mh_w_fc1=nn.Linear( config.n_heads * d_v,embedding_size)
            self.mh_w_qs2=nn.Linear(embedding_size, config.n_heads * d_k)
            self.mh_w_vs2=nn.Linear(embedding_size, config.n_heads * d_v)
            self.mh_w_fc2=nn.Linear( config.n_heads * d_v,embedding_size)
            shareweight=[self.mh_w_qs1,self.mh_w_vs1,self.mh_w_qs2,self.mh_w_vs2]
            for ly in shareweight:
                nn.init.normal_(ly.weight, mean=0, std=np.sqrt(2.0 / (embedding_size + d_k)))
            nn.init.kaiming_normal_(self.mh_w_fc2.weight)
            nn.init.kaiming_normal_(self.mh_w_fc1.weight)
        else:
            self.mh_w_qs1=None
            self.mh_w_vs1=None
            self.mh_w_fc1=None
            self.mh_w_qs2=None
            self.mh_w_vs2=None
            self.mh_w_fc2=None
        if config.select_kg ==True:
            #output_length~=inputlength/4
            self.Ws_kg=  nn.Sequential(
            nn.Conv1d(embedding_size,embedding_size*2,kernel_size=6,stride=2,padding=0),
            nn.Conv1d(embedding_size*2,embedding_size,kernel_size=3,stride=2,padding=0),
            nn.ReLU(inplace=True)
            )
        
        self.layer_stack_kg = nn.ModuleList([
            EncoderLayer(embedding_size,\
                config.hidden_size, config.n_heads, config.k_dims, config.v_dims, config.dropout,self.mh_w_qs1,self.mh_w_vs1,self.mh_w_fc1)
            for _ in range(config.n_layers)
        ])
        self.layer_stack_his= nn.ModuleList([
            EncoderLayer(embedding_size,\
                config.hidden_size, config.n_heads, config.k_dims, config.v_dims, config.dropout,self.mh_w_qs2,self.mh_w_vs2,self.mh_w_fc2)
            for _ in range(config.n_layers)
        ])
        #  enc_output = self.layer_norm(enc_output)
        # self.fc_out = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(embedding_size, config.hidden_size*2),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, char_his, kg):
        #history_embedded=[batchsize, seq=~106,embeddingsize=300]
        history_embedded= self.dropout(self.pos_embedding(self.char_embedding(char_his)))
        #kg_embed=[batchsize, seq=~103,embeddingsize=300]
        kg_embed=self.dropout(self.pos_embedding(self.char_embedding(kg)))
        for layer in self.layer_stack_kg:
            history_embedded, _ = layer(history_embedded)
        if self.Ws_kg !=None:
            info_embed=torch.cat((kg_embed,history_embedded),1)
            info_embed = info_embed.transpose(1, 2)
            kg_embed=self.Ws_kg(info_embed)
            kg_embed = info_embed.transpose(1, 2)
        #kg_embed(layer)=[b,s,embedding size] ->layer不变dim
        for layer in self.layer_stack_his:
            kg_embed, _ = layer(kg_embed)
        
        inputs_src=torch.cat((history_embedded,kg_embed),dim=1)
        # enc_outs = inputs_src.permute(0, 2, 1)
        # enc_outs = torch.sum(enc_outs, dim=-1)#enc_outs =[batchsize,seq,embedding]->[batchsize,embedding]
        # return self.fc_out(enc_outs)
        return inputs_src
class TransfomerDecoder(nn.Module):
    def __init__(self,config,embedding,pos_embedding,voc_size,ffn_dim=512):
        super(TransfomerDecoder, self).__init__()
        self.seq_embedding = embedding
        self.pos_embedding = pos_embedding
        if config.shareW==True:
            d_k=config.k_dims//config.n_heads
            d_v=config.k_dims//config.n_heads
            embedding_size=300
            self.mh_w_qs1=nn.Linear(embedding_size, config.n_heads * d_k)
            self.mh_w_vs1=nn.Linear(embedding_size, config.n_heads * d_v)
            self.mh_w_fc1=nn.Linear( config.n_heads * d_v,embedding_size)
            self.mh_w_qs2=nn.Linear(embedding_size, config.n_heads * d_k)
            self.mh_w_vs2=nn.Linear(embedding_size, config.n_heads * d_v)
            self.mh_w_fc2=nn.Linear( config.n_heads * d_v,embedding_size)
            shareweight=[self.mh_w_qs1,self.mh_w_vs1,self.mh_w_qs2,self.mh_w_vs2]
            for ly in shareweight:
                nn.init.normal_(ly.weight, mean=0, std=np.sqrt(2.0 / (embedding_size + d_k)))
            nn.init.kaiming_normal_(self.mh_w_fc2.weight)
            nn.init.kaiming_normal_(self.mh_w_fc1.weight)
        else:
            self.mh_w_qs1=None
            self.mh_w_vs1=None
            self.mh_w_fc1=None
            self.mh_w_qs2=None
            self.mh_w_vs2=None
            self.mh_w_fc2=None    
        self.decoder_layers = nn.ModuleList([DecoderLayer(300, config.k_dims, config.v_dims, \
            config.n_heads, ffn_dim, config.dropout,self.mh_w_qs1,self.mh_w_vs1,self.mh_w_fc1,\
                self.mh_w_qs2,self.mh_w_vs2,self.mh_w_fc2) for _ in range(config.n_layers)])
        self.final_transformer_linear = nn.Linear(300, voc_size, bias=False)
        self.final_transformer_softmax = nn.Softmax(dim=2)

    def forward(self, decoder_inputs, enc_output, context_attn_mask=None):
        #enc_output[B L E]
        #output:[B L E]
        output=self.pos_embedding(self.seq_embedding(decoder_inputs))
        # padding_mask requires that seq_k和seq_q的形状都是[B,L]
        #self_attention_padding_mask=[B,L,L]
        self_attention_padding_mask = padding_mask(decoder_inputs)
        seq_mask = sequence_mask(decoder_inputs)
        #self_attn_mask=[B,L,L]
        self_attn_mask = (torch.gt((self_attention_padding_mask.float() + seq_mask.float()), 0)).float()
        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
            output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
        
        #transformer forward:
        #context_attn_mask = padding_mask(tgt_seq, src_seq)
        #output, enc_self_attn = self.encoder(src_seq, src_len)
        #output, dec_self_attn, ctx_attn = self.decoder( tgt_seq, tgt_len, output, context_attn_mask)
        #output = self.linear(output)
        #output = self.softmax(output)
        # return output, enc_self_attn, dec_self_attn, ctx_attn
        output = self.final_transformer_linear(output)
        output = self.final_transformer_softmax(output)

        return output, self_attentions, context_attentions
