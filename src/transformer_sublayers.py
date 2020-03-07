# -*- encoding: utf-8 -*-
#' '
#@file_name    :transformer_sublayers.py
#@description    :
#@time    :2020/02/22 16:23:02
#@author    :Cindy, xd Zhang 
#@version   :0.1
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def residual(sublayer_fn,x):
	return sublayer_fn(x)+x
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
#    context_attn_mask = padding_mask(tgt_seq, src_seq)
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
class PositionwiseFeedForward(nn.Module):
    '''
        前馈神经网络
    '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        '''

        :param d_in:    输入维度
        :param d_hid:   隐藏层维度
        :param dropout:
        '''
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_normal = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        #output=[batch,x, d_in]
        output = self.layer_normal(output + residual)
        return output
class MultiHeadAttention(nn.Module):
    '''
        “多头”注意力模型
    '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout,mh_w_qs=None,mh_w_vs=None,mh_w_fc=None):
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
        if mh_w_qs!=None:self.w_qs=mh_w_qs
        else:
            self.w_qs = nn.Linear(d_model, n_head * d_k)
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # self.w_ks = nn.Linear(d_model, n_head * d_k)
        if mh_w_vs!=None:self.w_vs=mh_w_vs
        else:
            self.w_vs = nn.Linear(d_model, n_head * d_v)
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.layer_normal = nn.LayerNorm(d_model)
        if mh_w_fc!=None:self.fc=mh_w_fc
        else:
            self.fc = nn.Linear(n_head * d_v, d_model)
            nn.init.kaiming_normal_(self.fc.weight)

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
        k = self.w_qs(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        #enc+tgt 时:q=tgt[b,head,seqT,Dq];k=enc[b,head,seqE,Dq]
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        #
        output, attn = self.attention(q, k, v, mask=None)
        # (n_heads * batch_size) * lq * dv
        output = output.view(n_head, sz_b, len_q, d_v)
        # batch_size * len_q * (n_heads * dv)
        output= output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_normal(output + residual)
        return output, attn
class EncoderLayer(nn.Module):
    '''编码层'''
    def __init__(self, embedding_size, hidden_size, n_head, d_k, d_v, dropout,mh_w_qs=None,mh_w_vs=None,mh_w_fc=None):
        '''

        :param embedding_size: 模型输入维度==voc embedding_size
        :param hidden_size: 前馈神经网络隐层维度
        :param n_head:  多头注意力
        :param d_k:     键向量
        :param d_v:     值向量
        :param dropout:
        '''
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, embedding_size, d_k, d_v, dropout=dropout,mh_w_qs=mh_w_qs,mh_w_vs=mh_w_vs,mh_w_fc=mh_w_fc)
        self.pos_ffn = PositionwiseFeedForward(embedding_size, hidden_size, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        '''

        :param enc_input:
        :param non_pad_mask:
        :param slf_attn_mask:
        :return:
        '''
        #enc_input=[batchsize,seq,embeddingsize]
        #enc_output=[batchsize,seq,embeddingsize]
        #enc_slf_attn=[batchsize,head,seq,seq]
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn
class DecoderLayer(nn.Module):

    def __init__(self, model_dim, d_k, d_v, n_head=8, ffn_dim=2048, dropout=0.0,mh_w_qs1=None,mh_w_vs1=None,mh_w_fc1=None,mh_w_qs2=None,mh_w_vs2=None,mh_w_fc2=None):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, model_dim, d_k, d_v, dropout=dropout,mh_w_qs=mh_w_qs1,mh_w_vs=mh_w_vs1,mh_w_fc=mh_w_fc1)
        self.enc_attn = MultiHeadAttention(n_head, model_dim, d_k, d_v, dropout=dropout,mh_w_qs=mh_w_qs2,mh_w_vs=mh_w_vs2,mh_w_fc=mh_w_fc2)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout=dropout)

    def forward(self,dec_inputs,enc_outputs,self_attn_mask=None,context_attn_mask=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.slf_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.enc_attn(dec_output,enc_outputs, enc_outputs,context_attn_mask)

        # decoder's output, or context
        #hiddensize即model_dim;  feed_forward 输出dim为 ffn_dim
        dec_output = self.feed_forward(dec_output)
        
        return dec_output, self_attention, context_attention
