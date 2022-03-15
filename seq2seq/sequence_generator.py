import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceGeneratorStd(nn.Module):
    def __init__(self, model, beam_size: int, max_length: int, pad_idx = 0, sos_idx = 1, eos_idx = 2):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.max_length = max_length
        self.begin_token = self.sos_idx


    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)[:, -1, :]

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob[:, ::self.beam_size, :].contiguous()
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx     

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output = encoder_output.index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data,):
        input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence, \
            target_lengths, agent_positions, target_positions = batch_data
        input_lengths = torch.tensor(input_lengths).to(input_sequence).view(-1, 1)
        batch_size, seq_len = input_sequence.size()

        curr_device = input_sequence.device

        ## encoder forward
        encoded_input = self.model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)
        projected_keys_visual = self.model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = self.model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = self.model.attention_decoder.initialize_hidden(
            self.model.tanh(self.model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(input_sequence.device).long()

        new_hidden1 = hidden[0].index_select(1, new_order)
        new_hidden2 = hidden[1].index_select(1, new_order)
        new_hidden = (new_hidden1, new_hidden2)
        new_projected_keys_textual = projected_keys_textual.index_select(1, new_order)
        new_projected_keys_visual = projected_keys_visual.index_select(1, new_order)
        print(input_lengths)
        print(type(input_lengths))
        print(input_lengths.size())
        new_input_lengths = input_lengths.index_select(1, new_order)



        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 2).to(curr_device).fill_(self.sos_idx).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(input_sequence.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]
        unfinished_sents = len(finished)

        for position in range(self.max_length):
            decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = self.model.decode_input(
                target_token=generated_tokens, hidden=new_hidden, encoder_outputs=new_projected_keys_textual,
                input_lengths=new_input_lengths, encoded_situations=new_projected_keys_visual)
            
            output = F.log_softmax(output, dim=-1)
            output[:, self.pad_idx] = -math.inf
            output[:, self.sos_idx] = -math.inf
            
            if position >= self.max_length - 1:
                output[:] = -math.inf
                output[:, self.eos_idx] = 0
                generated_tokens[:, position + 1] = self.eos_idx
                break
            if position >= 1:
                eos_mask = generated_tokens[:, position].eq(self.eos_idx)
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                output[eos_mask] = -math.inf
                output[eos_mask, self.eos_idx] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(output, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)


            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs

            # print("beam search words: ", candidate_idxs)
            # print("beam search beam idx:", cand_bbsz_idx)
            # print("updated sequence: ")
            # print(generated_tokens[:, :position + 2])
            # pause = input("??")

            scores[:, position + 1] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.eos_idx)
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:]


class SequenceGenerator(nn.Module):
    def __init__(self, model, beam_size: int, max_length: int, pad_idx = 0, sos_idx = 1, eos_idx = 2):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.max_length = max_length
        self.begin_token = self.sos_idx


    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob[:, ::self.beam_size, :].contiguous()
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx     

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output = encoder_output.index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data, required_length_list = [], force_align = True):
        input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence, \
            target_lengths, agent_positions, target_positions = batch_data
        input_lengths = torch.tensor(input_lengths).to(input_sequence)
        batch_size, seq_len = input_sequence.size()

        curr_device = input_sequence.device

        ## encoder forward
        encoded_input = self.model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)
        projected_keys_visual = self.model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = self.model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # print(projected_keys_textual.size())
        # print(projected_keys_visual.size())

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = self.model.attention_decoder.initialize_hidden(
            self.model.tanh(self.model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))

        # print(hidden[0].size())

        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(input_sequence.device).long()

        new_hidden1 = hidden[0].index_select(1, new_order)
        new_hidden2 = hidden[1].index_select(1, new_order)
        new_hidden = (new_hidden1, new_hidden2)
        new_projected_keys_textual = projected_keys_textual.index_select(1, new_order)
        new_projected_keys_visual = projected_keys_visual.index_select(0, new_order)
        new_input_lengths = input_lengths.index_select(0, new_order)



        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(input_sequence.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]

        decode_dict = {}

        for position in range(self.max_length):
            # decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = self.model.decode_input(
                target_token=generated_tokens[:, position], hidden=new_hidden, encoder_outputs=new_projected_keys_textual,
                input_lengths=new_input_lengths, encoded_situations=new_projected_keys_visual)
            # new_hidden = hidden
            # output = F.softmax(output, dim=-1)

            output[:, self.pad_idx] = -math.inf
            output[:, self.sos_idx] = -math.inf

            if force_align:
                output[:, self.eos_idx] = -math.inf
            output = self.log_probs(output)

            
            if position >= self.max_length:
                output[:] = -math.inf
                output[:, self.eos_idx] = 0
                generated_tokens[:, position + 1] = self.eos_idx
                break
            # if position >= 1:
                # eos_mask = generated_tokens[:, position].eq(self.eos_idx)
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                # output[eos_mask] = -math.inf
                # output[eos_mask, self.eos_idx] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(output, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)

            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            new_hidden1 = hidden[0].index_select(1, cand_bbsz_idx)
            new_hidden2 = hidden[1].index_select(1, cand_bbsz_idx)
            new_hidden = (new_hidden1, new_hidden2)

            # new_projected_keys_textual = new_projected_keys_textual.index_select(1, cand_bbsz_idx)
            # new_projected_keys_visual = new_projected_keys_visual.index_select(0, cand_bbsz_idx)
            # new_input_lengths = new_input_lengths.index_select(0, cand_bbsz_idx)

            if position + 1 in required_length_list:
                decode_dict[position + 1] = generated_tokens[:, 1:position + 2].detach().clone().view(batch_size, self.beam_size, -1)

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs



            scores[:, position] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.eos_idx)
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:], decode_dict

class SequenceGeneratorStat(nn.Module):
    def __init__(self, model, beam_size: int, max_length: int, pad_idx = 0, sos_idx = 1, eos_idx = 2):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.max_length = max_length
        self.begin_token = self.sos_idx


    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob[:, ::self.beam_size, :].contiguous()
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)
        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)
        return top_scores, word_idx, beam_idx     

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output = encoder_output.index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data, required_length_list = [], force_align = True):
        input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence, \
            target_lengths, agent_positions, target_positions = batch_data
        input_lengths = torch.tensor(input_lengths).to(input_sequence)
        batch_size, seq_len = input_sequence.size()
        target_len = target_sequence.size(1)

        curr_device = input_sequence.device

        ## encoder forward
        encoded_input = self.model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)
        projected_keys_visual = self.model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = self.model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # print(projected_keys_textual.size())
        # print(projected_keys_visual.size())

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = self.model.attention_decoder.initialize_hidden(
            self.model.tanh(self.model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))

        # print(hidden[0].size())

        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(input_sequence.device).long()

        new_hidden1 = hidden[0].index_select(1, new_order)
        new_hidden2 = hidden[1].index_select(1, new_order)
        new_hidden = (new_hidden1, new_hidden2)
        new_projected_keys_textual = projected_keys_textual.index_select(1, new_order)
        new_projected_keys_visual = projected_keys_visual.index_select(0, new_order)
        new_input_lengths = input_lengths.index_select(0, new_order)



        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(input_sequence.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]

        decode_dict = {}
        batch_ground_truth_list_before = [[] for _ in range(batch_size)]
        batch_ground_truth_list_after = [[] for _ in range(batch_size)]

        for position in range(self.max_length):
            if position > target_len - 2:
                break
            # decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = self.model.decode_input(
                target_token=generated_tokens[:, position], hidden=new_hidden, encoder_outputs=new_projected_keys_textual,
                input_lengths=new_input_lengths, encoded_situations=new_projected_keys_visual)
            # new_hidden = hidden
            # output = F.softmax(output, dim=-1)
            before_reweight_score = F.softmax(output, dim = -1)

            output[:, self.pad_idx] = -math.inf
            output[:, self.sos_idx] = -math.inf

            if force_align:
                output[:, self.eos_idx] = -math.inf
            after_reweight_score = F.softmax(output, dim = -1)

            before_reweight_score = before_reweight_score.view(batch_size, self.beam_size, -1)
            after_reweight_score = after_reweight_score.view(batch_size, self.beam_size, -1)

            for idx in range(batch_size):
                curr_ground_truth = target_sequence[idx, position + 1]
                if curr_ground_truth in [self.pad_idx, self.sos_idx, self.eos_idx]:
                    continue
                before_prob = before_reweight_score[idx,0,curr_ground_truth].item()
                after_prob = after_reweight_score[idx,0,curr_ground_truth].item()
                batch_ground_truth_list_before[idx].append(before_prob)
                batch_ground_truth_list_after[idx].append(after_prob)


            output = self.log_probs(output)

            
            if position >= self.max_length:
                output[:] = -math.inf
                output[:, self.eos_idx] = 0
                generated_tokens[:, position + 1] = self.eos_idx
                break
            candidate_scores, candidate_idxs, candidate_beam_idxs = self.beam_search(output, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)

            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            new_hidden1 = hidden[0].index_select(1, cand_bbsz_idx)
            new_hidden2 = hidden[1].index_select(1, cand_bbsz_idx)
            new_hidden = (new_hidden1, new_hidden2)

            # new_projected_keys_textual = new_projected_keys_textual.index_select(1, cand_bbsz_idx)
            # new_projected_keys_visual = new_projected_keys_visual.index_select(0, cand_bbsz_idx)
            # new_input_lengths = new_input_lengths.index_select(0, cand_bbsz_idx)

            if position + 1 in required_length_list:
                decode_dict[position + 1] = generated_tokens[:, 1:position + 2].detach().clone().view(batch_size, self.beam_size, -1)

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs



            scores[:, position] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.eos_idx)
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:], decode_dict, batch_ground_truth_list_before, batch_ground_truth_list_after


class SequenceGeneratorV2(nn.Module):
    def __init__(self, model, beam_size: int, times, max_length: int, pad_idx = 0, sos_idx = 1, eos_idx = 2):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.times = times
        self.max_length = max_length
        self.begin_token = self.sos_idx


    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob[:, ::self.beam_size, :].contiguous()
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        if position == 0:
            top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)   
            all_pred = top_pred 
        else:
            top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1) 
            all_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size * self.times, dim = -1)

        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)

        all_scores, all_indices = all_pred
        all_beam_idx = all_indices // vocab_num
        all_word_idx = torch.fmod(all_indices, vocab_num)
        return top_scores, word_idx, beam_idx, all_word_idx, all_beam_idx

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output = encoder_output.index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data, required_length_list = [], force_align = True):
        input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence, \
            target_lengths, agent_positions, target_positions = batch_data
        input_lengths = torch.tensor(input_lengths).to(input_sequence)
        batch_size, seq_len = input_sequence.size()

        curr_device = input_sequence.device

        ## encoder forward
        encoded_input = self.model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)
        projected_keys_visual = self.model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = self.model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # print(projected_keys_textual.size())
        # print(projected_keys_visual.size())

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = self.model.attention_decoder.initialize_hidden(
            self.model.tanh(self.model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))

        # print(hidden[0].size())

        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(input_sequence.device).long()

        new_hidden1 = hidden[0].index_select(1, new_order)
        new_hidden2 = hidden[1].index_select(1, new_order)
        new_hidden = (new_hidden1, new_hidden2)
        new_projected_keys_textual = projected_keys_textual.index_select(1, new_order)
        new_projected_keys_visual = projected_keys_visual.index_select(0, new_order)
        new_input_lengths = input_lengths.index_select(0, new_order)



        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        prefix_tokens = torch.zeros(batch_size, self.beam_size * self.times, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()


        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(input_sequence.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]

        decode_dict = {}

        for position in range(self.max_length):
            # decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = self.model.decode_input(
                target_token=generated_tokens[:, position], hidden=new_hidden, encoder_outputs=new_projected_keys_textual,
                input_lengths=new_input_lengths, encoded_situations=new_projected_keys_visual)
            # new_hidden = hidden
            # output = F.softmax(output, dim=-1)
            output[:, self.pad_idx] = -math.inf
            output[:, self.sos_idx] = -math.inf

            if force_align:
                output[:, self.eos_idx] = -math.inf
            output = self.log_probs(output)            
            if position >= self.max_length:
                output[:] = -math.inf
                output[:, self.eos_idx] = 0
                generated_tokens[:, position + 1] = self.eos_idx
                break
            # if position >= 1:
                # eos_mask = generated_tokens[:, position].eq(self.eos_idx)
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                # output[eos_mask] = -math.inf
                # output[eos_mask, self.eos_idx] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs, all_candidate_idxs, all_beam_idxs = self.beam_search(output, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)

            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            # prefix_tokens = generated_tokens.view(batch_size, self.beam_size, -1).index_select(dim = 1, index = all_beam_idxs)
            generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
            if position > 1:
                for batch_idx in range(batch_size):
                    for beam_idx in range(all_beam_idxs.size(1)):
                        curr_beam = all_beam_idxs[batch_idx][beam_idx]
                        curr_token_id = all_candidate_idxs[batch_idx][beam_idx]
                        prefix_tokens[batch_idx, beam_idx, :position + 1] = generated_tokens[batch_idx, curr_beam, :position + 1]
                        prefix_tokens[batch_idx, beam_idx, position + 1] = curr_token_id
            else:
                prefix_tokens[:,:self.beam_size,:] = generated_tokens

            generated_tokens = generated_tokens.view(batch_size * self.beam_size, -1)

            new_hidden1 = hidden[0].index_select(1, cand_bbsz_idx)
            new_hidden2 = hidden[1].index_select(1, cand_bbsz_idx)
            new_hidden = (new_hidden1, new_hidden2)

            # new_projected_keys_textual = new_projected_keys_textual.index_select(1, cand_bbsz_idx)
            # new_projected_keys_visual = new_projected_keys_visual.index_select(0, cand_bbsz_idx)
            # new_input_lengths = new_input_lengths.index_select(0, cand_bbsz_idx)

            if position + 1 in required_length_list:
                # decode_dict[position + 1] = generated_tokens[:, 1:position + 2].detach().clone().view(batch_size, self.beam_size, -1)
                if position + 1 == 1:
                    decode_dict[position + 1] = prefix_tokens[:, :self.beam_size, 1:position + 2].detach().clone()
                else:
                    decode_dict[position + 1] = prefix_tokens[:, :, 1:position + 2].detach().clone()

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs



            scores[:, position] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.eos_idx)
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:], decode_dict

class SequenceGeneratorV3(nn.Module):
    def __init__(self, model, beam_size: int, times, max_length: int, pad_idx = 0, sos_idx = 1, eos_idx = 2):
        super().__init__()
        self.model = model
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.beam_size = beam_size
        self.times = times
        self.max_length = max_length
        self.begin_token = self.sos_idx


    def log_probs(self, logits):
        return F.log_softmax(logits, dim = -1)

    def beam_search(self, predict_prob, scores, generated_tokens, position):
        '''
        predict_prob: batch_size * beam_size, vocab_num
        scores: batch_size, beam_size, max_length 

        add predict prob with prob score   first reshape the predict_prob into batch_size, beam_size, 1   then broadcast
        '''
        total_item, vocab_num = predict_prob.size()
        batch_size = total_item // self.beam_size
        # predict_prob = predict_prob.view(batch_size, 1, vocab_num).repeat(1, self.beam_size, 1)
        predict_prob = predict_prob.view(batch_size, self.beam_size, vocab_num)
        if position == 0:
            predict_prob = predict_prob[:, ::self.beam_size, :].contiguous()
        else:
            predict_prob = predict_prob + scores[:,:, position - 1].unsqueeze(-1)    ## batch_size, beam_size, vocab_num

        if position == 0:
            top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1)   
            all_pred = top_pred 
        else:
            top_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size, dim = -1) 
            all_pred = torch.topk(predict_prob.view(batch_size, -1), k = self.beam_size * self.times, dim = -1)

        top_scores, top_indices = top_pred
        beam_idx = top_indices // vocab_num
        word_idx = torch.fmod(top_indices, vocab_num)

        all_scores, all_indices = all_pred
        all_beam_idx = all_indices // vocab_num
        all_word_idx = torch.fmod(all_indices, vocab_num)
        return top_scores, word_idx, beam_idx, all_word_idx, all_beam_idx

    def expand_encoder_output(self, encoder_output, new_order):
        encoder_output = encoder_output.index_select(0, new_order)
        return encoder_output

    def forward(self, batch_data, required_length_list = [], force_align = True):
        input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence, \
            target_lengths, agent_positions, target_positions = batch_data
        input_lengths = torch.tensor(input_lengths).to(input_sequence)
        batch_size, seq_len = input_sequence.size()

        curr_device = input_sequence.device

        ## encoder forward
        encoded_input = self.model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)
        projected_keys_visual = self.model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = self.model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # print(projected_keys_textual.size())
        # print(projected_keys_visual.size())

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = self.model.attention_decoder.initialize_hidden(
            self.model.tanh(self.model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))

        # print(hidden[0].size())

        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        ## expand the encoder_out to shape [batch_size * beam_size, seq_len, hidden_dim]
        new_order = torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1)
        new_order = new_order.to(input_sequence.device).long()

        new_hidden1 = hidden[0].index_select(1, new_order)
        new_hidden2 = hidden[1].index_select(1, new_order)
        new_hidden = (new_hidden1, new_hidden2)
        new_projected_keys_textual = projected_keys_textual.index_select(1, new_order)
        new_projected_keys_visual = projected_keys_visual.index_select(0, new_order)
        new_input_lengths = input_lengths.index_select(0, new_order)



        ## start beam search
        generated_tokens = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()
        scores = torch.zeros(batch_size * self.beam_size, self.max_length + 1).to(curr_device)

        prefix_tokens = torch.zeros(batch_size, self.beam_size * self.times, self.max_length + 1).to(curr_device).fill_(self.sos_idx).long()


        bbsz_offsets = (
                    (torch.arange(0, batch_size) * self.beam_size)
                    .unsqueeze(1)
                    .type_as(generated_tokens)
                    .to(input_sequence.device)
                )


        final_scores = torch.zeros(batch_size * self.beam_size).to(curr_device)

        generated_tokens[:,0] = self.begin_token
        finished_flag = torch.zeros(batch_size * self.beam_size).to(curr_device).eq(-1)
        finished = [False for i in range(batch_size)]

        decode_dict = {}

        for position in range(self.max_length):
            # decoder_output = self.model.forward_decoder(generated_tokens[:, :position + 1], new_encoder_out)
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = self.model.decode_input(
                target_token=generated_tokens[:, position], hidden=new_hidden, encoder_outputs=new_projected_keys_textual,
                input_lengths=new_input_lengths, encoded_situations=new_projected_keys_visual)
            # new_hidden = hidden
            # output = F.softmax(output, dim=-1)
            output = self.log_probs(output)
            output[:, self.pad_idx] = -math.inf
            output[:, self.sos_idx] = -math.inf

            if force_align:
                output[:, self.eos_idx] = -math.inf
            
            if position >= self.max_length:
                output[:] = -math.inf
                output[:, self.eos_idx] = 0
                generated_tokens[:, position + 1] = self.eos_idx
                break
            # if position >= 1:
                # eos_mask = generated_tokens[:, position].eq(self.eos_idx)
                # print("tokens at position ", position, ": ", generated_tokens[:, :position])
                # print(eos_mask)
                # output[eos_mask] = -math.inf
                # output[eos_mask, self.eos_idx] = 0
            # predict_probs[finished_flag, self.tgt_dict.pad()] = math.inf
            # print(predict_probs[finished_flag, self.tgt_dict.pad()])
            candidate_scores, candidate_idxs, candidate_beam_idxs, all_candidate_idxs, all_beam_idxs = self.beam_search(output, scores.view(batch_size, self.beam_size, -1), generated_tokens, position)


            cand_bbsz_idx = candidate_beam_idxs + bbsz_offsets    ## batch， beam, 对应到generated token中的位置
            cand_bbsz_idx = cand_bbsz_idx.view(-1)

            generated_tokens = generated_tokens[cand_bbsz_idx]
            generated_tokens[:, position + 1] = candidate_idxs.view(-1)

            # prefix_tokens = generated_tokens.view(batch_size, self.beam_size, -1).index_select(dim = 1, index = all_beam_idxs)
            generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
            if position > 1:
                for batch_idx in range(batch_size):
                    for beam_idx in range(all_beam_idxs.size(1)):
                        curr_beam = all_beam_idxs[batch_idx][beam_idx]
                        curr_token_id = all_candidate_idxs[batch_idx][beam_idx]
                        prefix_tokens[batch_idx, beam_idx, :position + 1] = generated_tokens[batch_idx, curr_beam, :position + 1]
                        prefix_tokens[batch_idx, beam_idx, position + 1] = curr_token_id
            else:
                prefix_tokens[:,:self.beam_size,:] = generated_tokens

            generated_tokens = generated_tokens.view(batch_size * self.beam_size, -1)

            new_hidden1 = hidden[0].index_select(1, cand_bbsz_idx)
            new_hidden2 = hidden[1].index_select(1, cand_bbsz_idx)
            new_hidden = (new_hidden1, new_hidden2)

            # new_projected_keys_textual = new_projected_keys_textual.index_select(1, cand_bbsz_idx)
            # new_projected_keys_visual = new_projected_keys_visual.index_select(0, cand_bbsz_idx)
            # new_input_lengths = new_input_lengths.index_select(0, cand_bbsz_idx)

            if position + 1 in required_length_list:
                # decode_dict[position + 1] = generated_tokens[:, 1:position + 2].detach().clone().view(batch_size, self.beam_size, -1)
                if position + 1 == 1:
                    decode_dict[position + 1] = prefix_tokens[:, :self.beam_size, 1:position + 2].detach().clone()
                else:
                    decode_dict[position + 1] = prefix_tokens[:, :, 1:position + 2].detach().clone()

            # generated_tokens[cand_bbsz_idx, position + 1] = candidate_idxs



            scores[:, position] = candidate_scores.view(-1)
            final_scores = scores[:, position + 1]
            # final_scores[new_eos_mask] = scores[new_eos_mask, position]  ## use the score before <eos>
            
            finished_flag = generated_tokens[:, position + 1].eq(self.eos_idx)
            
            if finished_flag.sum() == batch_size * self.beam_size:
                break
            if position >= self.max_length:
                break
        
        batch_scores = final_scores.view(batch_size, self.beam_size)
        best_beam = torch.argmax(batch_scores, dim = -1)
        generated_tokens = generated_tokens.view(batch_size, self.beam_size, -1)
        best_sentences = torch.stack([generated_tokens[i, best_beam[i]] for i in range(len(best_beam))], dim = 0)
        return best_sentences[:, 1:], decode_dict

