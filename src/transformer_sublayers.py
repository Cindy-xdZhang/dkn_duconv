# -*- encoding: utf-8 -*-
#' '
#@file_name    :transformer_sublayers.py
#@description    :
#@time    :2020/02/22 16:23:02
#@author    :Cindy, xd Zhang 
#@version   :0.1
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
def get_attn_pad_mask(seq_q, seq_k):
    #[B,LQ]  B,LQ,E * B,E,LK->B,LQ,LK 
    #[B,LK]
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k
def padding_mask(seq_q):
	# seq_q的形状是[B,L]
    # padding_mask 为shape [B, L, L],seq_q为0（pad)的地方x则对应的[Bi,:,x]为1
    #sample:
    #seq_q=tensor([[  1.,  44.,  23.,   2.,   0.,   0.,   0.], 
    #[  1., 424., 323., 422.,   2.,   0.,   0.]]) 
    # mask=tensor([[[0, 0, 0, 0, 1, 1, 1],
    # [0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1, 1],
    #  [0, 0, 0, 0, 1, 1, 1]],
    # [[0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 0, 1, 1]]], dtype=torch.uint8)
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_q.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L, L]
    return pad_mask
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask
class Embeddings(nn.Module):
    def __init__(self, vocab,d_model ):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid=300, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2.0 *(hid_j //2) / d_hid  ) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 1, -1e9)#mask==0??

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class MultiHeadAttention(nn.Module):
    '''
        “多头”注意力模型
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        '''

        :param n_head: “头”数
        :param d_model: 输入维度
        :param d_k: 键向量维度
        :param d_v: 值向量维度
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()
        d_k=d_k//n_head
        d_v=d_v//n_head
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 产生 查询向量q，键向量k， 值向量v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_normal = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        '''
        计算多头注意力
        :param q: 用于产生  查询向量
        :param k: 用于产生  键向量
        :param v:  用于产生 值向量
        :param mask:
        :return:
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        #batchsize,seq_lens,nhead,d_k
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        #enc+tgt 时:q=tgt[b,head,seqT,Dq];k=enc[b,head,seqE,Dq]
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        #
        output, attn = self.attention(q, k, v, mask=mask)
        # (n_heads * batch_size) * lq * dv
        output = output.view(n_head, sz_b, len_q, d_v)
        # batch_size * len_q * (n_heads * dv)
        output= output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_normal(output + residual)
        return output, attn
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_normal = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual=x
        middel=self.dropout(self.w_2(F.relu(self.w_1(x))))
        return  self.layer_normal(residual+middel)
class Decoder_layer(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, Embeddingsize,n_head,d_k,d_v,d_hidden,dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.enc_dec_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.FForward = PositionwiseFeedForward(Embeddingsize, d_hidden, dropout=dropout)
    def forward(self, dec_input,self_attn_mask,enc_out,enc_dec_mask):
        #X=B,L
        # slf_attn_mask=padding_mask(Embedding_x)
        outputDec, attn=self.self_attention(dec_input,dec_input,dec_input,self_attn_mask)
        output, attn=self.enc_dec_attention(outputDec,enc_out,enc_out,enc_dec_mask)
        output=self.FForward(output)
        return output
class Encoder_layer(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, Embeddingsize,n_head,d_k,d_v,d_hidden,dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_head, Embeddingsize, d_k, d_v, dropout)
        self.FForward = PositionwiseFeedForward(Embeddingsize, d_hidden, dropout=dropout)
    def forward(self, x,slf_attn_mask):
        #X=B,L
        # slf_attn_mask=padding_mask(Embedding_x)
        output, attn=self.self_attention(x,x,x,slf_attn_mask)
        output=self.FForward(output)
        return output