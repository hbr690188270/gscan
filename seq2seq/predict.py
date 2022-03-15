from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Iterator
import time
import json

from seq2seq.helpers import sequence_accuracy, strip_pad, strip_white
from seq2seq.gSCAN_dataset import GroundedScanDataset, GroundedScanDatasetPad
from seq2seq.sequence_generator import SequenceGenerator, SequenceGeneratorStat, SequenceGeneratorV2

from tqdm import tqdm

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def predict_and_save(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path

def predict(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_and_save_beamsearch(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_with_beamsearch(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += len(output_sequence)
                for j in range(len(output_sequence)):
                    accuracy = sequence_accuracy(output_sequence[j], target_sequence[j].tolist()[1:-1])
                    input_str_sequence = dataset.array_to_sentence(input_sequence[j].tolist(), vocabulary="input")
                    input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    target_str_sequence = dataset.array_to_sentence(target_sequence[j].tolist(), vocabulary="target")
                    target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    output_str_sequence = dataset.array_to_sentence(output_sequence[j], vocabulary="target")
                    output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                                   "derivation": derivation_spec,
                                   "target": target_str_sequence, "situation": situation_spec,
                                   "attention_weights_input": attention_weights_commands,
                                   "attention_weights_situation": attention_weights_situations,
                                   "accuracy": accuracy,
                                   "exact_match": True if accuracy == 100 else False,
                                   "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path

def predict_with_beamsearch(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()
    generator = SequenceGenerator(model, beam_size = 5, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx)
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    for batch in data_iterator:
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break

        decode_output = generator(batch)
        i += 1

        hyp_list = []
        for i in range(decode_output):
            hyp = strip_pad(decode_output[i], pad_idx, eos_idx).detach().cpu().numpy()
            hyp_list.append(hyp)

        yield (batch[0], None, None, hyp_list, batch[5],
               None, None, 0)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_and_save_sampling(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_sampling(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path

def predict_sampling(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        # Iteratively decode the output.
        contexts_situation = []
        # hidden = model.attention_decoder.initialize_hidden(
        #     model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        attention_weights_commands = []
        attention_weights_situations = []
        all_outputs = []
        for sampling_idx in range(100):
            token = torch.tensor([sos_idx], dtype=torch.long, device=device)
            output_sequence = []
            hidden = model.attention_decoder.initialize_hidden(
                model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
            decoding_iteration = 0

            while token != eos_idx and decoding_iteration <= max_decoding_steps:
                (output, hidden, context_situation, attention_weights_command,
                attention_weights_situation) = model.decode_input(
                    target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                    input_lengths=input_lengths, encoded_situations=projected_keys_visual)
                output = F.softmax(output, dim=-1)

                prob = output.view(-1)
                token = torch.multinomial(prob, 1)
                # print("prob: ", prob)
                # print("select: ", token)

                output_sequence.append(token.data.item())
                decoding_iteration += 1

            print(output_sequence)
            if output_sequence[-1] == eos_idx:
                output_sequence.pop()
            all_outputs.append(output_sequence)
        


        auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, all_outputs, target_sequence,
               situation)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_beamsearch_disc_prev(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()
    generator = SequenceGenerator(model, beam_size = 5, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx)
    generator.max_length = 20
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    result = []
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:

        batch_size = input_sequence.size(0)
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        batch = (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions)



        required_length_list = [i for i in range(16, 21)]

        # for max_len in [16,17,18,19,20]:
        # for max_len in [x for x in range(5,15)]:

        batch_hyps = []

        decode_output, decode_dict = generator(batch, required_length_list = required_length_list)

        for batch_id in range(batch_size):
            hyp_list = []
            for length in required_length_list:
                decode_res = decode_dict[length][batch_id]
                for j in range(len(decode_res)):
                    hyp = strip_pad(decode_res[j], pad_idx, eos_idx).detach().cpu().numpy()
                    hyp_list.append(hyp)
                # print(len(hyp))
                # print(hyp)
            batch_hyps.append(hyp_list)
        
        i += batch_size


        # yield (batch[0], None, None, hyp_list, batch[5],
        #        None, None, 0)
        for batch_id in range(batch_size):
            result.append([input_sequence[batch_id], derivation_spec[batch_id], situation_spec[batch_id], batch_hyps[batch_id], target_sequence[batch_id],
               situation[batch_id]])

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))
    return result

def predict_beamsearch_disc(data_batch, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, beam_size = 5, max_length = 20, min_length = 16) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    # start_time = time.time()
    generator = SequenceGenerator(model, beam_size = beam_size, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx)
    generator.max_length = max_length
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    result = []
    input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,\
         target_lengths, agent_positions, target_positions = data_batch

    batch_size = input_sequence.size(0)
    batch = (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
        target_lengths, agent_positions, target_positions)



    required_length_list = [i for i in range(min_length, max_length + 1)]

    # for max_len in [16,17,18,19,20]:
    # for max_len in [x for x in range(5,15)]:

    batch_hyps = []

    decode_output, decode_dict = generator(batch, required_length_list = required_length_list)

    for batch_id in range(batch_size):
        hyp_list = []
        for length in required_length_list:
            decode_res = decode_dict[length][batch_id]
            for j in range(len(decode_res)):
                hyp = strip_pad(decode_res[j], pad_idx, eos_idx).detach().cpu().numpy()
                hyp_list.append(hyp)
            # print(len(hyp))
            # print(hyp)
        batch_hyps.append(hyp_list)
    
    i += batch_size


    # yield (batch[0], None, None, hyp_list, batch[5],
    #        None, None, 0)
    for batch_id in range(batch_size):
        result.append([input_sequence[batch_id], derivation_spec[batch_id], situation_spec[batch_id], batch_hyps[batch_id], target_sequence[batch_id],
            situation[batch_id]])

    # elapsed_time = time.time() - start_time
    # logging.info("Predicted for {} examples.".format(i))
    # logging.info("Done predicting in {} seconds.".format(elapsed_time))
    return result

def predict_beamsearch_stat(data_batch, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, beam_size = 5, max_length = 20) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    # start_time = time.time()
    generator = SequenceGeneratorStat(model, beam_size = beam_size, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx)
    generator.max_length = max_length
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    result = []
    input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,\
         target_lengths, agent_positions, target_positions = data_batch

    batch_size = input_sequence.size(0)
    batch = (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
        target_lengths, agent_positions, target_positions)



    required_length_list = [i for i in range(16, max_length + 1)]

    # for max_len in [16,17,18,19,20]:
    # for max_len in [x for x in range(5,15)]:

    batch_hyps = []

    decode_output, decode_dict, prob_before, prob_after = generator(batch, required_length_list = required_length_list)

    # for batch_id in range(batch_size):
    #     hyp_list = []
    #     for length in required_length_list:
    #         decode_res = decode_dict[length][batch_id]
    #         for j in range(len(decode_res)):
    #             hyp = strip_pad(decode_res[j], pad_idx, eos_idx).detach().cpu().numpy()
    #             hyp_list.append(hyp)
    #         # print(len(hyp))
    #         # print(hyp)
    #     batch_hyps.append(hyp_list)
    
    # i += batch_size


    # for batch_id in range(batch_size):
    #     result.append([input_sequence[batch_id], derivation_spec[batch_id], situation_spec[batch_id], batch_hyps[batch_id], target_sequence[batch_id],
    #         situation[batch_id]])

    return prob_before, prob_after



def predict_beamsearch(data_batch, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, beam_size = 5, max_length = 20) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    # start_time = time.time()
    generator = SequenceGenerator(model, beam_size = beam_size, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx,)
    generator.max_length = max_length
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    result = []
    input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,\
         target_lengths, agent_positions, target_positions = data_batch

    batch_size = input_sequence.size(0)
    batch = (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
        target_lengths, agent_positions, target_positions)




    # for max_len in [16,17,18,19,20]:
    # for max_len in [x for x in range(5,15)]:

    batch_hyps = []

    decode_output, decode_dict = generator(batch, required_length_list = [], force_align = False)

    for batch_id in range(batch_size):
        hyp_list = []
        decode_res = decode_output[batch_id]
        hyp = strip_pad(decode_res, pad_idx, eos_idx).detach().cpu().numpy()
        hyp_list.append(hyp)
            # print(len(hyp))
            # print(hyp)
        batch_hyps.append(hyp_list)
    
    i += batch_size


    # yield (batch[0], None, None, hyp_list, batch[5],
    #        None, None, 0)
    for batch_id in range(batch_size):
        result.append([input_sequence[batch_id], derivation_spec[batch_id], situation_spec[batch_id], batch_hyps[batch_id], target_sequence[batch_id],
            situation[batch_id]])


    # elapsed_time = time.time() - start_time
    # logging.info("Predicted for {} examples.".format(i))
    # logging.info("Done predicting in {} seconds.".format(elapsed_time))
    return result

def predict_beamsearch_large(data_batch, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, beam_size = 5, max_length = 20, times = 5) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    # start_time = time.time()
    generator = SequenceGeneratorV2(model, beam_size = beam_size, times = times, max_length = max_decoding_steps, pad_idx=pad_idx, sos_idx = sos_idx, eos_idx=eos_idx)
    generator.max_length = max_length
    # Loop over the data.
    i = 0
    # for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
    #      target_lengths, agent_positions, target_positions) in data_iterator:
    result = []
    input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,\
         target_lengths, agent_positions, target_positions = data_batch

    batch_size = input_sequence.size(0)
    batch = (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
        target_lengths, agent_positions, target_positions)



    required_length_list = [i for i in range(21, max_length + 1)]

    # for max_len in [16,17,18,19,20]:
    # for max_len in [x for x in range(5,15)]:

    batch_hyps = []

    decode_output, decode_dict = generator(batch, required_length_list = required_length_list)

    for batch_id in range(batch_size):
        hyp_list = []
        for length in required_length_list:
            decode_res = decode_dict[length][batch_id]
            for j in range(len(decode_res)):
                hyp = strip_pad(decode_res[j], pad_idx, eos_idx).detach().cpu().numpy()
                hyp_list.append(hyp)
            # print(len(hyp))
            # print(hyp)
        batch_hyps.append(hyp_list)
    
    i += batch_size


    # yield (batch[0], None, None, hyp_list, batch[5],
    #        None, None, 0)
    for batch_id in range(batch_size):
        result.append([input_sequence[batch_id], derivation_spec[batch_id], situation_spec[batch_id], batch_hyps[batch_id], target_sequence[batch_id],
            situation[batch_id]])

    # elapsed_time = time.time() - start_time
    # logging.info("Predicted for {} examples.".format(i))
    # logging.info("Done predicting in {} seconds.".format(elapsed_time))
    return result

def predict_top5(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, max_length = 20) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    required_length_list = [i for i in range(16, max_length + 1)]
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_length:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            output = F.softmax(output, dim=-1)[0]

            output[eos_idx] = -99999


            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))



def predict_and_save_pad(dataset: GroundedScanDatasetPad, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_pad(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx, white_idx = dataset.white_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_pad(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, white_idx:int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        output_sequence = strip_white(output_sequence, white_idx)
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))



def predict_and_save_transformer(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_transformer(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_transformer(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        encoder_mask = input_sequence.eq(pad_idx)

        # # For efficiency
        # projected_keys_visual = model.visual_attention.key_layer(
        #     encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        # projected_keys_textual = model.textual_attention.key_layer(
        #     encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        # token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        token = torch.zeros([1, max_decoding_steps + 1], dtype = torch.long, device = device).fill_(pad_idx)
        token[0][0] = sos_idx
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token[0][decoding_iteration] != eos_idx and decoding_iteration < max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token[:, :decoding_iteration + 1], hidden=None, encoder_outputs=encoded_input["encoded_commands"]["encoder_outputs"],
                input_lengths=input_lengths, encoded_situations=encoded_input["encoded_situations"], encoder_mask = encoder_mask)
            output = F.log_softmax(output, dim=-1)
            pred_token = output.max(dim=-1)[1]
            output_sequence.append(pred_token.data[0].item())
            token[0][decoding_iteration + 1] = pred_token.data[0].item()
            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_with_target_length_transformer(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        encoder_mask = input_sequence.eq(pad_idx)

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        token = torch.zeros([1, max_decoding_steps + 1], dtype = torch.long, device = device).fill_(pad_idx)
        token[0][0] = sos_idx
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []

        while token[0][decoding_iteration] != eos_idx and decoding_iteration <= target_lengths[0] - 2:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token[:, :decoding_iteration + 1], hidden=None, encoder_outputs=encoded_input["encoded_commands"]["encoder_outputs"],
                input_lengths=input_lengths, encoded_situations=encoded_input["encoded_situations"], encoder_mask = encoder_mask)
            output = F.log_softmax(output, dim=-1)

            if decoding_iteration < target_lengths[0] - 2:
                output[0][eos_idx] = -99999
            elif decoding_iteration == target_lengths[0] - 2:
                output[0][eos_idx] = 9999

            pred_token = output.max(dim=-1)[1]
            output_sequence.append(pred_token.data[0].item())
            token[0][decoding_iteration + 1] = pred_token.data[0].item()

            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_and_save_noeos(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_with_target_length(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path

def predict_and_save_lenreg(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    alpha = kwargs['alpha']
    logger.info(f"length penalty {alpha}")
    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_with_length_reg(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx, alpha = alpha):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_and_save_noeos_transformer(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_with_target_length_transformer(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_with_target_length(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= target_lengths[0] - 2:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            output = F.log_softmax(output, dim=-1)
            
            if decoding_iteration < target_lengths[0] - 2:
                output[0][eos_idx] = -99999
            elif decoding_iteration == target_lengths[0] - 2:
                output[0][eos_idx] = 9999

            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_with_length_reg(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, alpha, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        target_lengths = torch.from_numpy(target_lengths).to(projected_keys_visual)

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= target_lengths[0] - 2:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            # output = F.log_softmax(output, dim=-1)
            output = F.softmax(output, dim = -1)
            length_penalty = alpha * F.relu(target_lengths - 2 - decoding_iteration)
            output[:, eos_idx] -= length_penalty
            # if decoding_iteration < target_lengths[0] - 2:
            #     output[0][eos_idx] = -99999
            # elif decoding_iteration == target_lengths[0] - 2:
            #     output[0][eos_idx] = 9999
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())

            # print(f"target len {target_lengths[0].item()}   decoding iter {decoding_iteration}   len_reg {length_penalty.item()}")
            # print(output[0])
            # print(output[0][eos_idx])

            # print(token.data[0].item())
            # print()
            # pause = input("??")

            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_length(data_iterator: Iterator, model: nn.Module,):
    for (input_batch, input_lengths, _, situation_batch, _, target_batch,
        target_lengths, agent_positions, target_positions) in data_iterator:

        is_best = False
        model.eval()
        # Forward pass.
        pred_lengths = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                    situations_input=situation_batch, target_batch=target_batch,
                                                    target_lengths=target_lengths)
        loss = model.get_loss(pred_lengths, target_lengths)
        gap = model.get_metrics(pred_lengths, target_lengths)

def predict_withlenbias(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        target_lengths = torch.from_numpy(target_lengths).to(projected_keys_visual).float()
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual, 
                target_lengths = target_lengths, curr_position = decoding_iteration)
            output = F.log_softmax(output, dim=-1)

            length_penalty = model.penalty_alpha * F.relu(target_lengths - 2 - decoding_iteration)
            output[:, eos_idx] -= length_penalty
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_withlenbias_and_save(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 attention_weights_commands, attention_weights_situations, position_accuracy) in predict_withlenbias(
                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "derivation": derivation_spec,
                               "target": target_str_sequence, "situation": situation_spec,
                               "attention_weights_input": attention_weights_commands,
                               "attention_weights_situation": attention_weights_situations,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               "position_accuracy":  position_accuracy})
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_and_save_eos_distribution(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                 eos_distribution, second_distribution) in predict_eos_distribution(
                #  eos_distribution, second_distribution) in predict_eos_distribution_v2(

                    dataset.get_data_iterator(batch_size=1), model=model, max_decoding_steps=max_decoding_steps,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx):
                i += 1
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
                input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
                input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
                target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
                output.append({"input": input_str_sequence, "prediction": output_str_sequence,
                               "target": target_str_sequence,
                                "eos_distribution": eos_distribution,
                                "second_distribution": second_distribution,
                               "accuracy": accuracy,
                               "exact_match": True if accuracy == 100 else False,
                               })
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path


def predict_eos_distribution(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        eos_distribution = []
        second_distribution = []
        while token != eos_idx and decoding_iteration <= target_lengths[0] - 2:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            # output = F.log_softmax(output, dim=-1)
            output = F.softmax(output, dim=-1)

            eos_distribution.append(output[0][eos_idx].item())

            if decoding_iteration < target_lengths[0] - 2:
                output[0][eos_idx] = -99999
                second_distribution.append(output.max(dim=-1)[0][0].item())

            elif decoding_iteration == target_lengths[0] - 2:
                output[0][eos_idx] = 9999


            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()

        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                eos_distribution, second_distribution,
               )

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_eos_distribution_v2(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_sequence, input_lengths, derivation_spec, situation, situation_spec, target_sequence,
         target_lengths, agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        encoded_input = model.encode_input(commands_input=input_sequence,
                                           commands_lengths=input_lengths,
                                           situations_input=situation)

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        eos_distribution = []
        second_distribution = []
        flag = True
        while token != eos_idx and decoding_iteration <= target_lengths[0] - 2:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths, encoded_situations=projected_keys_visual)
            # output = F.log_softmax(output, dim=-1)
            output = F.softmax(output, dim=-1)

            eos_distribution.append(output[0][eos_idx].item())

            if decoding_iteration < target_lengths[0] - 2 and output.max(dim=-1)[1][0].item() == eos_idx and flag:
                output[0][eos_idx] = -99999
                second_distribution.append(output.max(dim=-1)[0][0].item())
                flag = False

            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            # attention_weights_commands.append(attention_weights_command.tolist())
            # attention_weights_situations.append(attention_weights_situation.tolist())
            # contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            # attention_weights_commands.pop()
            # attention_weights_situations.pop()

        yield (input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence,
                eos_distribution, second_distribution,
               )

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))


def predict_disc(data_iterator: Iterator, model: nn.Module, max_examples_to_evaluate=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0
    for (input_batch, token_type_ids, input_lengths, derivation_spec, situation_batch, situation_spec, labels,
            agent_positions, target_positions) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i > max_examples_to_evaluate:
                break
        # Encode the input sequence.
        logits = model(commands_input=input_batch,
                            token_type_ids = token_type_ids,
                            commands_lengths=input_lengths,
                            situations_input=situation_batch,
                            )        
        pred_labels = torch.argmax(logits, dim = 1)
        # yield (input_batch, derivation_spec, situation_spec, logits, labels, pred_labels)
        yield (input_batch, logits, labels, pred_labels, input_lengths)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

def predict_and_save_disc(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str,
                     max_testing_examples=None, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    cfg = locals().copy()

    with open(output_file_path, mode='w') as outfile:
        output = []
        with torch.no_grad():
            i = 0
            for (input_sequence, derivation_spec, situation_spec, logits, labels, pred_labels) in predict_disc(
                    dataset.get_data_iterator(batch_size=200), model=model,):
                loss, acc, recall, precision, F1 = model.get_metrics(logits, labels)
                batch_size = input_sequence.size(0)
                for idx in range(batch_size):
                    i += 1
                    input_str_sequence = dataset.array_to_sentence(input_sequence[idx].tolist(), vocabulary="input")
                    input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
                    end_of_sen1 = -1
                    for i in range(len(input_str_sequence)):
                        if input_str_sequence == dataset.input_vocabulary.sos_token:
                            end_of_sen1 = i
                            break
                    source = input_str_sequence[:end_of_sen1]
                    target = input_str_sequence[end_of_sen1 + 1: ]

                    output.append({"input": source, "prediction": pred_labels[idx].item(),
                                "derivation": derivation_spec,
                                "target": target, "situation": situation_spec,
                                })
        logger.info("Wrote predictions for {} examples.".format(i))
        json.dump(output, outfile, indent=4)
    return output_file_path

