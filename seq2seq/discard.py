import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import torch
import numpy as np
import copy
import random

from GroundedScan.dataset import GroundedScan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)




class GroundedScanDatasetContrast(object):
    """
    Loads a GroundedScan instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, insertion = 'mid', aug_prob = 1.0,
                 white_portion = 0.5, aug_strategy = 'rand', max_white_num = 10,):
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory, k=k)
        if self.dataset._data_statistics.get("adverb_1"):
            logger.info("Verb-adverb combinations in training set: ")
            for adverb, items in self.dataset._data_statistics["train"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
            logger.info("Verb-adverb combinations in dev set: ")
            for adverb, items in self.dataset._data_statistics["dev"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
        self.image_dimensions = None
        self.image_channels = 3
        self.split = split
        self.directory = save_directory

        # Keeping track of data.
        self._examples = []
        self._input_lengths = []
        self._target_lengths = []
        self._labels = []

        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            self.target_vocabulary.add_sentence(['_'])
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")
        self.white_token = '_'
        self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token)

        common_dict = self.combine_dict(self.input_vocabulary, self.target_vocabulary)
        self.input_vocabulary = common_dict
        self.target_vocabulary = common_dict

        self.white_idx = self.target_vocabulary.word_to_idx(self.white_token)
        self.aug_prob = aug_prob
        self.white_portion = white_portion
        self.aug_num = 0
        self.insertion = insertion
        self.max_white_num = max_white_num
        self.aug_strategy = aug_strategy


    def combine_dict(self, source_vocab, target_vocab):
        source_vocab.combine_dict(target_vocab)
        return source_vocab


    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split)):
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random_permutation = np.random.permutation(len(self._examples))
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]
        self._examples = self._examples[random_permutation]
        self._labels = self._labels[random_permutation]

    def get_data_iterator(self, batch_size=10) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[dict],
                                                        torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param batch_size: how many examples to put in each batch.
        :param auxiliary_task: if true, also batches agent and target positions (flattened, so
        agent row * agent columns = agent_position)
        :return: tuple of input commands batch, corresponding input lengths, situation image batch,
        list of corresponding situation representations, target commands batch and corresponding target lengths.
        """
        for example_i in range(0, len(self._examples), batch_size):
            if example_i + batch_size > len(self._examples):
                batch_size = len(self._examples) - example_i
            examples = self._examples[example_i:example_i + batch_size]
            input_lengths = self._input_lengths[example_i:example_i + batch_size]
            target_lengths = self._target_lengths[example_i:example_i + batch_size]
            labels = self._labels[example_i:example_i + batch_size]

            # max_input_length = np.max(input_lengths)
            # max_target_length = np.max(target_lengths)
            input_lengths = input_lengths + target_lengths
            max_length = np.max(input_lengths) - 1

            input_batch = []
            token_type_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                input_x = torch.cat([example["input_tensor"], example['target_tensor'][:,1:]], dim = 1)
                sen1_len = example["input_tensor"].size(1)
                sen2_len = example['target_tensor'].size(1) - 1
                token_type_list = sen1_len * [0] + sen2_len * [1]
                token_type_tensor = torch.LongTensor(token_type_list).unsqueeze(0).to(device)

                to_pad_input = max_length - input_x.size(1)
                padded_input = torch.cat([
                    input_x,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                token_type_tensor = torch.cat([
                    token_type_tensor,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                assert token_type_tensor.size(1) == padded_input.size(1)
                # print("sen1: ", example["input_tensor"])
                # print("sen2: ",  example['target_tensor'])
                # print("final: ", padded_input)
                # print("segment: ", token_type_tensor)
                # pause = input("???")
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long, device=device),
                #     torch.zeros(int(to_pad_input), dtype=torch.long, device=devicedevice).unsqueeze(0)], dim=1) # TODO: change back
                input_batch.append(padded_input)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])
                token_type_batch.append(token_type_tensor)
            labels = torch.LongTensor(labels).to(device)

            yield (torch.cat(input_batch, dim=0), torch.cat(token_type_batch, dim = 0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch,
                   labels,
                   torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        logger.info("Converting dataset to tensors...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split, simple_situation_representation)):
            if max_examples:
                if len(self._examples) > max_examples:
                    self._input_lengths = np.array(self._input_lengths)
                    self._target_lengths = np.array(self._target_lengths)
                    self._examples = np.array(self._examples)
                    self._labels = np.array(self._labels)
                    logger.info(f"total augmentations: {self.aug_num}")    
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            situation_repr = example["situation_representation"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            #empty_example["equivalent_target_tensor"] = torch.tensor(equivalent_target_array, dtype=torch.long,
            #                                                         device=device).unsqueeze(dim=0)
            empty_example["situation_tensor"] = torch.tensor(situation_image, dtype=torch.float, device=device
                                                             ).unsqueeze(dim=0)
            empty_example["situation_representation"] = situation_repr
            empty_example["derivation_representation"] = example["derivation_representation"]
            empty_example["agent_position"] = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long,
                device=device).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long, device=device).unsqueeze(dim=0)
            

            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(target_array))
            self._examples.append(empty_example)
            self._labels.append(1)


            aug_target_array = self.aug_target_idx_list(target_array)
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(aug_target_array))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(aug_target_array, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(1)
            self.aug_num += 1

            contra1_target, contra2_target = self.contrast_creator.contrast_neg_num2(orig_command = target_commands)
            contra_target_array1 = self.sentence_to_array(contra1_target, 'target')
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(contra_target_array1))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(contra_target_array1, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(0)
            self.aug_num += 1            

            contra_target_array2 = self.sentence_to_array(contra2_target, 'target')
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(contra_target_array2))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(contra_target_array2, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(0)
            self.aug_num += 1      
            


        self._input_lengths = np.array(self._input_lengths)
        self._target_lengths = np.array(self._target_lengths)
        self._examples = np.array(self._examples)
        self._labels = np.array(self._labels)
        logger.info(f"total augmentations: {self.aug_num}")          

    def aug_target_idx_list(self, target_seq):
        target_len = len(target_seq) - 2
        if self.aug_strategy == 'fixed':
            expanded_len = int(target_len * self.white_portion)
            total_len = target_len + expanded_len
            if self.insertion == 'mid':
                selected_indices = np.random.choice(total_len, size = expanded_len, replace = False)
                expanded_array = [self.target_vocabulary.sos_idx]
                count = 0
                for idx in range(total_len):
                    if idx in selected_indices:
                        expanded_array.append(self.white_idx)
                        count += 1
                    else:
                        expanded_array.append(target_seq[1 + idx - count])
                expanded_array.append(self.target_vocabulary.eos_idx)
            elif self.insertion == 'start':
                selected_indices = np.arange(expanded_len)
                expanded_array = [self.target_vocabulary.sos_idx]
                count = 0
                for idx in range(total_len):
                    if idx in selected_indices:
                        expanded_array.append(self.white_idx)
                        count += 1
                    else:
                        expanded_array.append(target_seq[1 + idx - count])
                expanded_array.append(self.target_vocabulary.eos_idx)
            else:
                raise NotImplementedError
        elif self.aug_strategy == 'rand':
            expanded_len = np.random.randint(self.max_white_num) + 1
            total_len = target_len + expanded_len
            if self.insertion == 'mid':
                selected_indices = np.random.choice(total_len, size = expanded_len, replace = False)
                expanded_array = [self.target_vocabulary.sos_idx]
                count = 0
                for idx in range(total_len):
                    if idx in selected_indices:
                        expanded_array.append(self.white_idx)
                        count += 1
                    else:
                        expanded_array.append(target_seq[1 + idx - count])
                expanded_array.append(self.target_vocabulary.eos_idx)
            elif self.insertion == 'start':
                selected_indices = np.arange(expanded_len)
                expanded_array = [self.target_vocabulary.sos_idx]
                count = 0
                for idx in range(total_len):
                    if idx in selected_indices:
                        expanded_array.append(self.white_idx)
                        count += 1
                    else:
                        expanded_array.append(target_seq[1 + idx - count])
                expanded_array.append(self.target_vocabulary.eos_idx)
            else:
                raise NotImplementedError
        else:
            NotImplementedError      
        # pause = input("???")
        return expanded_array


    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
