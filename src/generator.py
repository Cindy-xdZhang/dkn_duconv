# -*- encoding: utf-8 -*-
#' '
#@file_name    :translator.py
#@description    :
#@time    :2020/03/24 14:07:48
#@author    :Cindy, xd Zhang 
#@version   :0.1
import torch
import torch.nn as nn
from network import TransfomerEncoder,TransfomerDecoder
class Beam(object):
    ''' Store the necessary info for beam search '''
    def __init__(self, size, use_cuda=False):
        self.size = size
        self.done = False

        self.tt = torch.cuda if use_cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(data_utils.PAD)]
        self.next_ys[0][0] = data_utils.BOS

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_lk + self.scores.unsqueeze(1).expand_as(word_lk)
        else:
            beam_lk = word_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True) # 2nd sort TODO

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened beam_size * tgt_vocab_size array, so calculate
        # which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == data_utils.EOS:
            self.done = True
            self.all_scores.append(self.scores)

        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[data_utils.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps))

        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        for j in range(len(self.prev_ks)-1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return hyp[::-1]

class Response_Generator(object):
    ''' Load with trained model and handel the beam search '''
    def __init__(self, opt,encoder,decoder):
        self.opt = opt
        self.encoder = encoder
        self.decoder = decoder
        self.prob_proj = nn.LogSoftmax(dim=-1)
        self.encoder.eval()
        self.decoder.eval()

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        history,knowledge,responses=src_batch["history"],src_batch["knowledge"],src_batch["response"]
        history = pad_sequence(history,batch_first=True, padding_value=0).to(network.Global_device)
        knowledge = pad_sequence(knowledge,batch_first=True, padding_value=0).to(network.Global_device)
        responses =pad_sequence(responses,batch_first=True, padding_value=0).to(network.Global_device)
        batch_size = history.size(0) # enc_inputs: [batch_size x src_len]
        beam_size = self.opt.beam_size
        # Encode
        enc_outputs, _ =  self.model.encode(enc_inputs,knowledge)

        # Repeat data for beam
        #B,L->B,3L->B*3,L
        enc_inputs = enc_inputs.repeat(1, beam_size).view(batch_size * beam_size, -1)
        #B,L,E->B,3L,E->B*3,L,E
        enc_outputs =  enc_outputs.repeat(1, beam_size, 1).view(
            batch_size * beam_size, enc_outputs.size(1), enc_outputs.size(2))

        # Prepare beams
        beams = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))
        }
        n_remaining_sents = batch_size

        # Decode
        for i in range(self.opt.max_decode_step):
            len_dec_seq = i + 1
            # Preparing decoded data_seq
            # size: [batch_size x beam_size x seq_len]
            dec_partial_inputs = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            # size: [batch_size * beam_size x seq_len]
            dec_partial_inputs = dec_partial_inputs.view(-1, len_dec_seq)
            # wrap into a Variable
            dec_partial_inputs = Variable(dec_partial_inputs, volatile=True)

            # Preparing decoded pos_seq
            # size: [1 x seq]
            # dec_partial_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0) # TODO:
            # # size: [batch_size * beam_size x seq_len]
            # dec_partial_pos = dec_partial_pos.repeat(n_remaining_sents * beam_size, 1)
            # # wrap into a Variable
            # dec_partial_pos = Variable(dec_partial_pos.type(torch.LongTensor), volatile=True)
            dec_partial_inputs_len = torch.LongTensor(n_remaining_sents,).fill_(len_dec_seq) # TODO: note
            dec_partial_inputs_len = dec_partial_inputs_len.repeat(beam_size)
            #dec_partial_inputs_len = Variable(dec_partial_inputs_len, volatile=True)

            if self.use_cuda:
                dec_partial_inputs = dec_partial_inputs.cuda()
                dec_partial_inputs_len = dec_partial_inputs_len.cuda()

            # Decoding
            dec_outputs, *_ = self.model.decode(dec_partial_inputs, dec_partial_inputs_len,
                                                enc_inputs, enc_outputs) # TODO:
            dec_outputs = dec_outputs[:,-1,:] # [batch_size * beam_size x d_model]
            dec_outputs = self.model.tgt_proj(dec_outputs)
            out = self.model.prob_proj(dec_outputs)

            # [batch_size x beam_size x tgt_vocab_size]
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx] # 해당 beam_idx 의 데이터가 실제 data 에서 몇번째 idx인지
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list: # all instances have finished their path to <eos>
                break

            # In this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list]) # TODO: fix

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                inst_idx_dim_size, *rest_dim_sizes = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, *rest_dim_sizes = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = (inst_idx_dim_size, *rest_dim_sizes)

                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.model_opt.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                return Variable(active_enc_info_data, volatile=True)

            enc_inputs = update_active_seq(enc_inputs, active_inst_idxs)
            enc_outputs = update_active_enc_info(enc_outputs, active_inst_idxs)

            # update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        # Return useful information
        all_hyp, all_scores = [], []
        n_best = self.opt.n_best

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]

        return all_hyp, all_scores
