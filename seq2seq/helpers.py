import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_mask(sequence_lengths: torch.LongTensor, max_len=None) -> torch.tensor:
    """
    Create a sequence mask that masks out all indices larger than some sequence length as defined by
    sequence_lengths entries.

    :param sequence_lengths: [batch_size] sequence lengths per example in batch
    :param max_len: int defining the maximum sequence length in the batch
    :return: [batch_size, max_len] boolean mask
    """
    if max_len is None:
        max_len = sequence_lengths.data.max()
    batch_size = sequence_lengths.size(0)
    sequence_range = torch.arange(0, max_len).long().to(device=device)

    # [batch_size, max_len]
    sequence_range_expand = sequence_range.unsqueeze(0).expand(batch_size, max_len)

    # [batch_size, max_len]
    seq_length_expand = (sequence_lengths.unsqueeze(1).expand_as(sequence_range_expand))

    # [batch_size, max_len](boolean array of which elements to include)
    return sequence_range_expand < seq_length_expand


def log_parameters(model: torch.nn.Module) -> {}:
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Total parameters: %d" % n_params)
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info("%s : %s" % (name, list(p.size())))


def sequence_accuracy(prediction: List[int], target: List[int]) -> float:
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        difference = len(prediction) - len(target)
        target.extend([-1] * difference)
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100

def sequence_accuracy_noeos(prediction: List[int], target: List[int]) -> float:
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        prediction = prediction[:len(target)]
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100


def strip_pad(tensor, pad, eos):
    # print(tensor)
    eos_idx = (tensor == eos).nonzero()
    if len(eos_idx) == 0:
        tensor = tensor
    else:
        tensor = tensor[:eos_idx[0][0]]
    return tensor[tensor.ne(pad)]

def strip_white(idx_list, white):
    # print(tensor)
    return [x for x in idx_list if x != white]

def cal_recall(ground_truth, pred_seq_list):
    for seq in pred_seq_list:
        acc = sequence_accuracy(seq, ground_truth)
        # print(seq)
        # print(ground_truth)
        # print()
        # pause = input("???")

        if acc == 100:
            return 1
    return 0


def entropy(logits, reduction = 'mean'):
    batch_size, class_num = logits.size()
    b = F.softmax(logits, dim = -1) * F.log_softmax(logits, dim = -1)
    b = -1.0 * torch.sum(b, dim = 1)
    b = torch.mean(b)
    return b

def stat_prob_list(batch_prob_list):
    max_len = max([len(x) for x in batch_prob_list])
    length_prob_dict = {x:[] for x in range(max_len)}
    for prob_list in batch_prob_list:
        for pos in range(len(prob_list)):
            prob = prob_list[pos]
            length_prob_dict[pos].append(prob)
    avg_probs = [np.mean(length_prob_dict[x]) for x in range(max_len)]
    return avg_probs


