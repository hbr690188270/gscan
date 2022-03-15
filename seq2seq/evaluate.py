from seq2seq.predict import predict, predict_pad, predict_transformer, predict_length, predict_withlenbias, predict_disc
from seq2seq.helpers import sequence_accuracy, sequence_accuracy_noeos
import torch
import torch.nn as nn
from typing import Iterator
from typing import Tuple
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score



def evaluate(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None, no_eos = False) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        if no_eos:
            accuracy = sequence_accuracy_noeos(output_sequence, target_sequence[0].tolist()[1:-1])
        else:
            accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))

def evaluate_pad(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, white_idx, max_examples_to_evaluate=None) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict_pad(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, white_idx = white_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))

def evaluate_transformer(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict_transformer(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))


def evaluate_length_pred(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None):
    
    l2_distance_list = []
    l1_distance_list = []
    target_len_list = []

    for (input_batch, input_lengths, _, situation_batch, _, target_batch,
            target_lengths, agent_positions, target_positions) in data_iterator:
        is_best = False

        # Forward pass.
        target_lengths = torch.from_numpy(target_lengths).to(input_batch).float()
        pred_length = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                        situations_input=situation_batch, target_batch=target_batch,
                                                        target_lengths=target_lengths)
        l2_loss = model.get_loss(pred_length, target_lengths, reduction = 'none').detach().cpu().numpy().tolist()
        l1_metric = model.get_metrics(pred_length, target_lengths, reduction = 'none').detach().cpu().numpy().tolist()

        l2_distance_list += l2_loss
        l1_distance_list += l1_metric
        target_len_list += target_lengths.cpu().numpy().tolist()
        # print(pred_length)
        # print(target_lengths)
        # pause = input("???")
    return np.mean(l2_distance_list), np.mean(l1_distance_list), l2_distance_list, l1_distance_list, target_len_list

def evaluate_withlen(data_iterator: Iterator, model: nn.Module, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    exact_match = 0
    for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target in predict_withlenbias(
            data_iterator=data_iterator, model=model, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
        if accuracy == 100:
            exact_match += 1
        accuracies.append(accuracy)
        target_accuracies.append(aux_acc_target)
    return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
            float(np.mean(np.array(target_accuracies))))

def evaluate_disc(data_iterator: Iterator, model: nn.Module, max_examples_to_evaluate=None, multi_class = False) -> Tuple[float, float, float]:
    accuracies = []
    target_accuracies = []
    corr_total = 0
    total = 0
    all_pred = []
    all_orig = []
    length_acc_dict = {}
    for input_batch, logits, labels, pred_labels, input_lengths in predict_disc(
            data_iterator=data_iterator, model=model, max_examples_to_evaluate=max_examples_to_evaluate):
        corr = (pred_labels == labels).sum().item()
        corr_flags = (pred_labels == labels)
        corr_total += corr
        total += input_batch.size(0)
        all_pred.append(pred_labels)
        all_orig.append(labels)

        # print("pred: ", pred_labels)
        # print("orig: ", labels)
    all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
    all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
    if multi_class:
        precision = precision_score(all_orig, all_pred, average = 'macro')
        recall = recall_score(all_orig, all_pred, average = 'macro')
        F1 = f1_score(all_orig, all_pred, average = 'macro')

    else:

        precision = precision_score(all_orig, all_pred)
        recall = recall_score(all_orig, all_pred)
        F1 = f1_score(all_orig, all_pred)
    accuracy = corr_total / total
    print(corr_total, total, accuracy)

    return accuracy, recall, precision, F1

