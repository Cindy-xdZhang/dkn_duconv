# -*- encoding: utf-8 -*-
#'''
#@file_name    :network.py
#@description    :
#@time    :2020/02/13 12:47:27
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import torch
import torch.nn as nn
import torch.nn.functional as F
Global_device="cpu"

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

