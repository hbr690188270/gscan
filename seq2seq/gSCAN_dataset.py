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

from torch.utils.data import Dataset

from GroundedScan.dataset import GroundedScan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def contains_word(self, word: str) -> bool:
        if self._word_to_idx[word] != 0:
            return True
        else:
            return False

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def combine_dict(self, vocab2):
        dict2 = vocab2.to_dict()['word_to_idx']
        new_words = list(dict2.keys())
        for word in new_words:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
                print("adding word: ", word)
                print("word idx: ", self._word_to_idx[word])
            # self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, max_len = 20, min_len = 0, no_eos = False):
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
        self.max_len = max_len
        self.min_len = min_len
        self.no_eos = no_eos

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

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
        self._examples = self._examples[random_permutation]
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]

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
            max_input_length = np.max(input_lengths)
            max_target_length = np.max(target_lengths)
            input_batch = []
            target_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                to_pad_input = max_input_length - example["input_tensor"].size(1)
                to_pad_target = max_target_length - example["target_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long, device=device),
                #     torch.zeros(int(to_pad_input), dtype=torch.long, device=devicedevice).unsqueeze(0)], dim=1) # TODO: change back
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

            yield (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                   target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

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
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue

            situation_repr = example["situation_representation"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            if self.no_eos:
                target_array = self.sentence_to_array(target_commands, vocabulary="target")[:-1]
            else:
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
            self._input_lengths = np.append(self._input_lengths, [len(input_array)])
            self._target_lengths = np.append(self._target_lengths, [len(target_array)])
            self._examples = np.append(self._examples, [empty_example])

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


class GroundedScanDatasetForTest(object):
    """
    Loads a GroundedScan instance from a specified location.
    add length constrain
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, max_len = 20, min_len = 0):
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
        self.max_len = max_len
        self.min_len = min_len

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

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
        self._examples = self._examples[random_permutation]
        self._target_lengths = self._target_lengths[random_permutation]
        self._input_lengths = self._input_lengths[random_permutation]

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
            max_input_length = np.max(input_lengths)
            max_target_length = np.max(target_lengths)
            input_batch = []
            target_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                to_pad_input = max_input_length - example["input_tensor"].size(1)
                to_pad_target = max_target_length - example["target_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long, device=device),
                #     torch.zeros(int(to_pad_input), dtype=torch.long, device=devicedevice).unsqueeze(0)], dim=1) # TODO: change back
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

            yield (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                   target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

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
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            #equivalent_target_commands = example["equivalent_target_command"]
            situation_image = example["situation_image"]
            if i == 0:
                self.image_dimensions = situation_image.shape[0]
                self.image_channels = situation_image.shape[-1]
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue
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
            self._input_lengths = np.append(self._input_lengths, [len(input_array)])
            self._target_lengths = np.append(self._target_lengths, [len(target_array)])
            self._examples = np.append(self._examples, [empty_example])

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


class GroundedScanDatasetPad(object):
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
        self.white_idx = self.target_vocabulary.word_to_idx(self.white_token)
        self.aug_prob = aug_prob
        self.white_portion = white_portion
        self.aug_num = 0
        self.insertion = insertion
        self.max_white_num = max_white_num
        self.aug_strategy = aug_strategy

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
            max_input_length = np.max(input_lengths)
            max_target_length = np.max(target_lengths)
            input_batch = []
            target_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            for example in examples:
                to_pad_input = max_input_length - example["input_tensor"].size(1)
                to_pad_target = max_target_length - example["target_tensor"].size(1)
                padded_input = torch.cat([
                    example["input_tensor"],
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                # padded_input = torch.cat([
                #     torch.zeros_like(example["input_tensor"], dtype=torch.long, device=device),
                #     torch.zeros(int(to_pad_input), dtype=torch.long, device=devicedevice).unsqueeze(0)], dim=1) # TODO: change back
                padded_target = torch.cat([
                    example["target_tensor"],
                    torch.zeros(int(to_pad_target), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                input_batch.append(padded_input)
                target_batch.append(padded_target)
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

            yield (torch.cat(input_batch, dim=0), input_lengths, derivation_representation_batch,
                   torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                   target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

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

            rand_num = np.random.uniform()
            if rand_num < self.aug_prob:
                aug_target_array = self.aug_target_idx_list(target_array)
                self._input_lengths.append(len(input_array))
                self._target_lengths.append(len(aug_target_array))
                empty_example = copy.deepcopy(empty_example)
                empty_example["target_tensor"] = torch.tensor(aug_target_array, dtype=torch.long, device=device).unsqueeze(
                                                                dim=0)
                self._examples.append(empty_example)
                self.aug_num += 1
        
        self._input_lengths = np.array(self._input_lengths)
        self._target_lengths = np.array(self._target_lengths)
        self._examples = np.array(self._examples)
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


class ContrastCreator():
    def __init__(self, vocabulary: Vocabulary, max_del = 5, max_add = 5, max_copy = 5, max_sub = 5, white_token = '_',
                    contrast_prob = None, use_white = False):
        self.vocab = vocabulary
        self.max_del = max_del
        self.max_add = max_add
        self.max_copy = max_copy
        self.max_sub = max_sub
        self.contrast_strategy = ['delete', 'insert', 'append', 'sub','copy_insert']
        if contrast_prob != None:
            self.contrast_prob = contrast_prob
        else:
            self.contrast_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        vocab_dict = self.vocab.to_dict()
        
        self.idx2word = vocab_dict['idx_to_word']
        self.word2idx = vocab_dict['word_to_idx']
        self.sos_token = vocab_dict['sos_token']
        self.eos_token = vocab_dict['eos_token']
        self.pad_token = vocab_dict['pad_token']
        self.white_token = white_token
        self.white_token_idx = self.word2idx[self.white_token]

        self.add_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]

        if use_white:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token,]]
        else:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]



    def contrast_neg_num2(self, orig_command):
        select_contra = np.random.choice(len(self.contrast_prob), size = 2, p = self.contrast_prob)
        contra1, contra2 = select_contra[0], select_contra[1]
        
        if contra1 == 0:
            if len(orig_command) == 1:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.delete_transform(orig_command)
        elif contra1 == 1:
            contra1_res = self.insert_transform(orig_command)
        elif contra1 == 2:
            contra1_res = self.append_transform(orig_command)
        elif contra1 == 3:
            contra1_res = self.sub_transform(orig_command)
        elif contra1 == 4:
            contra1_res = self.copy_transform(orig_command)
        else:
            raise NotImplementedError

        if contra2 == 0:
            if len(orig_command) == 1:
                contra2_res = self.delete_transform(orig_command)
            else:
                contra2_res= self.append_transform(orig_command)
        elif contra2 == 1:
            contra2_res = self.insert_transform(orig_command)
        elif contra2 == 2:
            contra2_res = self.append_transform(orig_command)
        elif contra2 == 3:
            contra2_res = self.sub_transform(orig_command)
        elif contra2 == 4:
            contra2_res = self.copy_transform(orig_command)
        else:
            raise NotImplementedError
        return contra1_res, contra2_res

    def contrast_neg_num1(self, orig_command):
        select_contra = np.random.choice(len(self.contrast_prob), size = 1, p = self.contrast_prob)
        
        if select_contra == 0:
            if len(orig_command) == 1:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.delete_transform(orig_command)
        elif select_contra == 1:
            contra1_res = self.insert_transform(orig_command)
        elif select_contra == 2:
            contra1_res = self.append_transform(orig_command)
        elif select_contra == 3:
            contra1_res = self.sub_transform(orig_command)
        elif select_contra == 4:
            contra1_res = self.copy_transform(orig_command)
        else:
            raise NotImplementedError

        return contra1_res


    def delete_transform(self, orig_command):
        orig_length = len(orig_command)
        del_num = np.random.randint(self.max_del) + 1
        if del_num >= orig_length:
            del_num = 1
        del_indicess = np.random.choice(orig_length, size = del_num, replace = False)
        new_command = [orig_command[i] for i in range(orig_length) if i not in del_indicess]
        
        # print("del transform")
        # print("del pos: ", del_indicess)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")
        return new_command

    def append_transform(self, orig_command):
        orig_length = len(orig_command)
        add_num = np.random.randint(self.max_add) + 1
        total_length = orig_length + add_num
        append_token_idxs = np.random.choice(len(self.add_token_list), size = add_num, replace = True)
        append_tokens = [self.add_token_list[x] for x in append_token_idxs]
        new_command = orig_command + append_tokens
        # print("append num: ", add_num)
        # print("append tokens: ", append_tokens)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command

    def insert_transform(self, orig_command):
        orig_length = len(orig_command)
        insert_num = np.random.randint(self.max_add) + 1
        total_length = orig_length + insert_num
        while True:
            insert_token_idxs = np.random.choice(len(self.insert_token_list), size = insert_num, replace = True)
            flag = False
            for idx in insert_token_idxs:
                if idx != self.white_token_idx:
                    flag = True
                    break
            if flag:
                break
        

        insert_tokens = [self.insert_token_list[x] for x in insert_token_idxs]
        new_command = insert_tokens + orig_command
        # print("insert num: ", insert_num)
        # print("insert tokens: ", insert_tokens)
        # print("insert idx: ", insert_token_idxs)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command
        
    def copy_transform(self, orig_command):
        orig_length = len(orig_command)
        copy_num = np.random.randint(self.max_copy) + 1
        prob = np.random.uniform()
        if prob > 0.5:
            rand_indice = orig_length
            copy_token_idx = np.random.choice(len(self.insert_token_list))
            copy_token = self.insert_token_list[copy_token_idx]
            copy_tokens = [copy_token for _ in range(copy_num)]
            new_command = orig_command[: rand_indice] + copy_tokens
        else:
            rand_indice = np.random.choice(orig_length)
            copy_tokens = [orig_command[rand_indice] for _ in range(copy_num)]
            new_command = orig_command[:rand_indice] + copy_tokens + orig_command[rand_indice:]
        # print("copy indice: ", rand_indice)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # print("copy: ", copy_tokens)
        # pause = input("???")
        return new_command
    
    def sub_transform(self, orig_command):
        orig_length = len(orig_command)
        sub_num = np.random.randint(self.max_sub) + 1
        if sub_num > orig_length:
            sub_num = 1
        sub_indices = np.random.choice(orig_length, size = sub_num, replace = False)
        sub_token_idxs = np.random.choice(len(self.add_token_list), size = sub_num, replace = True)
        sub_tokens = [self.add_token_list[x] for x in sub_token_idxs]

        # new_command = orig_command[:]
        # for i in range(len(sub_token_idxs)):
        #     position = sub_indices[i]
        #     token = sub_tokens[i]
        #     new_command[position] = token

        new_command = orig_command[:]
        for i in range(len(sub_indices)):
            position = sub_indices[i]
            orig_token = orig_command[position]
            token_candidates = [x for x in self.add_token_list if x != orig_token]
            sub_token_idx = np.random.choice(len(token_candidates))
            token = token_candidates[sub_token_idx]
            new_command[position] = token

        # print("sub num: ", sub_num)
        # print("sub indices: ", sub_indices)
        # print("sub tokens: ", sub_tokens)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command

class ContrastCreatorDelete():
    def __init__(self, vocabulary: Vocabulary, max_del = 5, max_add = 5, max_copy = 5, max_sub = 5, white_token = '_',
                    contrast_prob = None, use_white = False):
        self.vocab = vocabulary
        self.max_del = max_del
        self.max_add = max_add
        self.max_copy = max_copy
        self.max_sub = max_sub
        self.contrast_strategy = ['delete', 'insert', 'append', 'sub','copy_insert']
        if contrast_prob != None:
            self.contrast_prob = contrast_prob
        else:
            self.contrast_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        vocab_dict = self.vocab.to_dict()
        
        self.idx2word = vocab_dict['idx_to_word']
        self.word2idx = vocab_dict['word_to_idx']
        self.sos_token = vocab_dict['sos_token']
        self.eos_token = vocab_dict['eos_token']
        self.pad_token = vocab_dict['pad_token']
        self.white_token = white_token
        self.white_token_idx = self.word2idx[self.white_token]

        self.add_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]

        if use_white:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token,]]
        else:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]


    def contrast_neg_num1(self, orig_command):

        contra1_res = self.delete_transform(orig_command)

        return contra1_res


    def delete_transform(self, orig_command):
        orig_length = len(orig_command)
        min_del = orig_length - 14
        max_del = min_del + self.max_del
        del_num = np.random.randint(min_del, max_del)
        if del_num >= orig_length:
            del_num = 1
        del_indicess = np.random.choice(orig_length, size = del_num, replace = False)
        new_command = [orig_command[i] for i in range(orig_length) if i not in del_indicess]
        
        # print("del transform")
        # print("del pos: ", del_indicess)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")
        return new_command

class ContrastCreatorControl():
    def __init__(self, vocabulary: Vocabulary, max_del = 5, max_add = 5, max_copy = 5, max_sub = 5, white_token = '_',
                    contrast_prob = None, use_white = False):
        self.vocab = vocabulary
        self.max_del = max_del
        self.max_add = max_add
        self.max_copy = max_copy
        self.max_sub = max_sub
        self.contrast_strategy = ['delete', 'insert', 'append', 'sub','copy_insert']
        if contrast_prob != None:
            self.contrast_prob = contrast_prob
        else:
            self.contrast_prob = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        vocab_dict = self.vocab.to_dict()
        
        self.idx2word = vocab_dict['idx_to_word']
        self.word2idx = vocab_dict['word_to_idx']
        self.sos_token = vocab_dict['sos_token']
        self.eos_token = vocab_dict['eos_token']
        self.pad_token = vocab_dict['pad_token']
        self.white_token = white_token
        self.white_token_idx = self.word2idx[self.white_token]

        self.add_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]

        if use_white:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token,]]
        else:
            self.insert_token_list = [x for x in self.word2idx.keys() if x not in [self.sos_token, self.eos_token, self.pad_token, self.white_token]]


    def contrast_neg_num1(self, orig_command):
        select_contra = np.random.choice(len(self.contrast_prob), size = 1, p = self.contrast_prob)
        
        if select_contra == 0:
            if len(orig_command) == 1:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.delete_transform(orig_command)
        elif select_contra == 1:
            if len(orig_command) >= 15:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.insert_transform(orig_command)
        elif select_contra == 2:
            if len(orig_command) >= 15:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.append_transform(orig_command)
        elif select_contra == 3:
            contra1_res = self.sub_transform(orig_command)
        elif select_contra == 4:
            if len(orig_command) >= 15:
                contra1_res = self.sub_transform(orig_command)
            else:
                contra1_res = self.copy_transform(orig_command)
        else:
            raise NotImplementedError

        return contra1_res




    def delete_transform(self, orig_command):
        orig_length = len(orig_command)
        del_num = np.random.randint(self.max_del) + 1
        if del_num >= orig_length:
            del_num = 1
        del_indicess = np.random.choice(orig_length, size = del_num, replace = False)
        new_command = [orig_command[i] for i in range(orig_length) if i not in del_indicess]
        
        # print("del transform")
        # print("del pos: ", del_indicess)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")
        return new_command

    def append_transform(self, orig_command):
        orig_length = len(orig_command)
        max_add = min(15 - orig_length, self.max_add)
        add_num = np.random.randint(max_add) + 1
        total_length = orig_length + add_num
        append_token_idxs = np.random.choice(len(self.add_token_list), size = add_num, replace = True)
        append_tokens = [self.add_token_list[x] for x in append_token_idxs]
        new_command = orig_command + append_tokens
        # print("append num: ", add_num)
        # print("append tokens: ", append_tokens)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command

    def insert_transform(self, orig_command):
        orig_length = len(orig_command)
        max_add = min(15 - orig_length, self.max_add)
        insert_num = np.random.randint(max_add) + 1
        total_length = orig_length + insert_num
        while True:
            insert_token_idxs = np.random.choice(len(self.insert_token_list), size = insert_num, replace = True)
            flag = False
            for idx in insert_token_idxs:
                if idx != self.white_token_idx:
                    flag = True
                    break
            if flag:
                break
        

        insert_tokens = [self.insert_token_list[x] for x in insert_token_idxs]
        new_command = insert_tokens + orig_command
        # print("insert num: ", insert_num)
        # print("insert tokens: ", insert_tokens)
        # print("insert idx: ", insert_token_idxs)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command
        
    def copy_transform(self, orig_command):
        orig_length = len(orig_command)
        max_copy = min(15 - orig_length, self.max_copy)
        copy_num = np.random.randint(max_copy) + 1
        prob = np.random.uniform()
        if prob > 0.5:
            rand_indice = orig_length
            copy_token_idx = np.random.choice(len(self.insert_token_list))
            copy_token = self.insert_token_list[copy_token_idx]
            copy_tokens = [copy_token for _ in range(copy_num)]
            new_command = orig_command[: rand_indice] + copy_tokens
        else:
            rand_indice = np.random.choice(orig_length)
            copy_tokens = [orig_command[rand_indice] for _ in range(copy_num)]
            new_command = orig_command[:rand_indice] + copy_tokens + orig_command[rand_indice:]
        # print("copy indice: ", rand_indice)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # print("copy: ", copy_tokens)
        # pause = input("???")
        return new_command
    
    def sub_transform(self, orig_command):
        orig_length = len(orig_command)
        sub_num = np.random.randint(self.max_sub) + 1
        if sub_num > orig_length:
            sub_num = 1
        sub_indices = np.random.choice(orig_length, size = sub_num, replace = False)
        sub_token_idxs = np.random.choice(len(self.add_token_list), size = sub_num, replace = True)
        sub_tokens = [self.add_token_list[x] for x in sub_token_idxs]

        # new_command = orig_command[:]
        # for i in range(len(sub_token_idxs)):
        #     position = sub_indices[i]
        #     token = sub_tokens[i]
        #     new_command[position] = token

        new_command = orig_command[:]
        for i in range(len(sub_indices)):
            position = sub_indices[i]
            orig_token = orig_command[position]
            token_candidates = [x for x in self.add_token_list if x != orig_token]
            sub_token_idx = np.random.choice(len(token_candidates))
            token = token_candidates[sub_token_idx]
            new_command[position] = token

        # print("sub num: ", sub_num)
        # print("sub indices: ", sub_indices)
        # print("sub tokens: ", sub_tokens)
        # print("orig: ", orig_command)
        # print("cont: ", new_command)
        # pause = input("???")

        return new_command


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


class GroundedScanDatasetContrastV2(object):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens


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
        self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token, use_white = False)

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


            # aug_target_array = self.aug_target_idx_list(target_array)
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(aug_target_array))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(aug_target_array, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(1)
            # self.aug_num += 1

            contra1_target = self.contrast_creator.contrast_neg_num1(orig_command = target_commands)
            contra_target_array1 = self.sentence_to_array(contra1_target, 'target')
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(contra_target_array1))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(contra_target_array1, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(0)
            self.aug_num += 1            

            # contra_target_array2 = self.sentence_to_array(contra2_target, 'target')
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(contra_target_array2))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(contra_target_array2, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(0)
            # self.aug_num += 1      
            


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

class GroundedScanDatasetContrastV3(object):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens


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
        self.contrast_creator = ContrastCreatorDelete(self.target_vocabulary, white_token = self.white_token, use_white = False)

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


            # aug_target_array = self.aug_target_idx_list(target_array)
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(aug_target_array))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(aug_target_array, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(1)
            # self.aug_num += 1

            contra1_target = self.contrast_creator.contrast_neg_num1(orig_command = target_commands)
            contra_target_array1 = self.sentence_to_array(contra1_target, 'target')
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(contra_target_array1))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(contra_target_array1, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(0)
            self.aug_num += 1            

            # contra_target_array2 = self.sentence_to_array(contra2_target, 'target')
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(contra_target_array2))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(contra_target_array2, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(0)
            # self.aug_num += 1      
            


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

class GroundedScanDatasetContrastV4(object):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens


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
        self.contrast_creator = ContrastCreatorControl(self.target_vocabulary, white_token = self.white_token, use_white = False)

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


            # aug_target_array = self.aug_target_idx_list(target_array)
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(aug_target_array))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(aug_target_array, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(1)
            # self.aug_num += 1

            contra1_target = self.contrast_creator.contrast_neg_num1(orig_command = target_commands)
            contra_target_array1 = self.sentence_to_array(contra1_target, 'target')
            self._input_lengths.append(len(input_array))
            self._target_lengths.append(len(contra_target_array1))
            empty_example = copy.deepcopy(empty_example)
            empty_example["target_tensor"] = torch.tensor(contra_target_array1, dtype=torch.long, device=device).unsqueeze(
                                                            dim=0)
            self._examples.append(empty_example)
            self._labels.append(0)
            self.aug_num += 1            

            # contra_target_array2 = self.sentence_to_array(contra2_target, 'target')
            # self._input_lengths.append(len(input_array))
            # self._target_lengths.append(len(contra_target_array2))
            # empty_example = copy.deepcopy(empty_example)
            # empty_example["target_tensor"] = torch.tensor(contra_target_array2, dtype=torch.long, device=device).unsqueeze(
            #                                                 dim=0)
            # self._examples.append(empty_example)
            # self._labels.append(0)
            # self.aug_num += 1      
            


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

class MultipleChoiceGScan(object):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens
    for each batch, randomly sample contrast data
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, insertion = 'mid', aug_prob = 1.0,
                 white_portion = 0.5, aug_strategy = 'rand', max_white_num = 10, contrast_size = 20, 
                 contrast_from_batch_size = 10, length_control = False,
                 max_len = 100, min_len = 0,
                 ):
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
        if length_control:
            self.contrast_creator = ContrastCreatorControl(self.target_vocabulary, white_token = self.white_token, use_white = False)
        else:
            self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token, use_white = False)


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
        self.contrast_size = contrast_size
        self.contrast_from_batch_size = contrast_from_batch_size
        self.max_len = max_len
        self.min_len = min_len

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

    # def tensor_to_string(self, batch_tensor):
    #     action_list = []
    #     for i in range(batch_tensor.size(0)):
    #         action = self.array_to_sentence(batch_tensor[i][1:-1], vocabulary = 'target')
    #         action_list.append(action)
    #     return action_list

    def tensor_to_string(self, tensor):
        action = self.array_to_sentence(tensor[1:-1], vocabulary = 'target')
        return action

    def contrast_aug_manually(self, batch_actions):
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        for i in range(batch_size):
            contrasted_actions = []
            orig_action = batch_actions[i]
            for j in range(self.contrast_size):
                contra_action = self.contrast_creator.contrast_neg_num1(orig_command = orig_action)
                contrasted_actions.append(contra_action)
            batch_contrasted_actions.append(contrasted_actions)
        return batch_contrasted_actions



    def contrast_aug_from_batch(self, batch_actions):
        contrast_size = self.contrast_from_batch_size
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        copy_batch_actions = copy.deepcopy(batch_actions)
        for i in range(batch_size):
            curr_contrasted_actions = []
            random.shuffle(copy_batch_actions)
            for j in range(batch_size):
                if batch_actions[i] != copy_batch_actions[j]:
                    curr_contrasted_actions.append(copy_batch_actions[j])
                if len(curr_contrasted_actions) >= contrast_size:
                    break
            curr_contrast_size = len(curr_contrasted_actions)
            if curr_contrast_size < contrast_size:
                contrast_size = curr_contrast_size
            batch_contrasted_actions.append(curr_contrasted_actions[:contrast_size])
        batch_contrasted_actions = [x[:contrast_size] for x in batch_contrasted_actions]
        return batch_contrasted_actions

    def tokenize_target(self, action_list):
        array_list = [self.sentence_to_array(x, vocabulary = 'target') for x in action_list]
        array_length = [len(x) for x in array_list]
        return array_list, array_length


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

            batch_target_actions = []
            for data_idx in range(len(examples)):
                curr_example = examples[data_idx]
                orig_target = curr_example['target_tensor'].squeeze(0)  ## seq_len
                orig_target_action = self.tensor_to_string(orig_target)
                batch_target_actions.append(orig_target_action)

            contrast_actions = self.contrast_aug_manually(batch_target_actions)  ## list of list:  batch_size    contrast_size 
            contrast_actions_from_batch = self.contrast_aug_from_batch(batch_target_actions)
            combined_contrast_actions = []
            for idx in range(len(contrast_actions)):
                combined = contrast_actions[idx] + contrast_actions_from_batch[idx]
                combined_contrast_actions.append(combined)

            batch_contrast_array = []
            batch_contrast_length = []
            for item in combined_contrast_actions:
                array_list, length_list = self.tokenize_target(item)
                batch_contrast_array.append(array_list)
                batch_contrast_length.append(length_list)
            batch_contrast_length = np.stack(batch_contrast_length, axis = 0)
            batch_max_len = np.max(batch_contrast_length, axis = 1)
            
            all_target_len = np.stack([batch_max_len, target_lengths], axis = 0)
            max_target_len = np.max(all_target_len, axis = 0)


            max_length = np.max(input_lengths + max_target_len) - 1


            # for example in examples:
            input_batch = []
            token_type_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            labels_batch = []

            for idx in range(len(examples)):
                example = examples[idx]
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
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

                input_list = [padded_input]
                segment_list = [token_type_tensor]
                # input_batch.append(padded_input)
                # token_type_batch.append(token_type_tensor)
                for aug_idx in range(len(batch_contrast_array[idx])):
                    curr_contrast_target = batch_contrast_array[idx][aug_idx]
                    curr_target_tensor = torch.tensor(curr_contrast_target, dtype=torch.long, device=device).unsqueeze(dim = 0)
                    input_x = torch.cat([example["input_tensor"], curr_target_tensor[:, 1:]], dim = 1)
                    sen2_len = curr_target_tensor.size(1) - 1
                    token_type_list = sen1_len * [0] + sen2_len * [1]
                    token_type_tensor = torch.LongTensor(token_type_list).unsqueeze(0).to(device)

                    to_pad_input = max_length - input_x.size(1)
                    padded_input = torch.cat([
                        input_x,
                        torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                    token_type_tensor = torch.cat([
                        token_type_tensor,
                        torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                    assert token_type_tensor.size(1) == padded_input.size(1), "%d -- %d "%(token_type_tensor.size(1), padded_input.size(1))
                    
                    input_list.append(padded_input)
                    # situation_batch.append(example["situation_tensor"])
                    # situation_representation_batch.append(example["situation_representation"])
                    # derivation_representation_batch.append(example["derivation_representation"])
                    # agent_positions_batch.append(example["agent_position"])
                    # target_positions_batch.append(example["target_position"])
                    segment_list.append(token_type_tensor)        
                input_list = torch.cat(input_list, dim = 0)
                segment_list = torch.cat(segment_list, dim = 0)
                rand_order = torch.randperm(input_list.size(0), device = device)
                input_list = input_list[rand_order]
                segment_list = segment_list[rand_order]
                label = torch.argmin(rand_order)  ## find wherer 0 is
                labels_batch.append(label)
                input_batch.append(input_list)
                token_type_batch.append(segment_list)
            
            labels = torch.LongTensor(labels_batch).to(device)

            yield (torch.stack(input_batch, dim=0), torch.stack(token_type_batch, dim = 0), input_lengths, derivation_representation_batch,
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
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue
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

        self._input_lengths = np.array(self._input_lengths)
        self._target_lengths = np.array(self._target_lengths)
        self._examples = np.array(self._examples)
        logger.info(f"total augmentations: {self.aug_num}")          

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

class GScanNLI(Dataset):
    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, insertion = 'mid', aug_prob = 1.0,
                 white_portion = 0.5, aug_strategy = 'rand', max_white_num = 10, contrast_size = 20, 
                 contrast_from_batch_size = 10, length_control = False,
                 max_examples = None, max_len = 100, min_len = 0,
                 ):
        super().__init__()
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
        if length_control:
            self.contrast_creator = ContrastCreatorControl(self.target_vocabulary, white_token = self.white_token, use_white = False)
        else:
            self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token, use_white = False)


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
        self.contrast_size = contrast_size
        self.contrast_from_batch_size = contrast_from_batch_size
        self.max_len = max_len
        self.min_len = min_len

        self.read_dataset(max_examples = max_examples)

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


    def tensor_to_string(self, tensor):
        action = self.array_to_sentence(tensor[1:-1], vocabulary = 'target')
        return action

    def contrast_aug_manually(self, batch_actions):
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        for i in range(batch_size):
            contrasted_actions = []
            orig_action = batch_actions[i]
            for j in range(self.contrast_size):
                contra_action = self.contrast_creator.contrast_neg_num1(orig_command = orig_action)
                contrasted_actions.append(contra_action)
            batch_contrasted_actions.append(contrasted_actions)
        return batch_contrasted_actions

    def contrast_aug_from_batch(self, batch_actions):
        contrast_size = self.contrast_from_batch_size
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        copy_batch_actions = copy.deepcopy(batch_actions)
        for i in range(batch_size):
            curr_contrasted_actions = []
            random.shuffle(copy_batch_actions)
            for j in range(batch_size):
                if batch_actions[i] != copy_batch_actions[j]:
                    curr_contrasted_actions.append(copy_batch_actions[j])
                if len(curr_contrasted_actions) >= contrast_size:
                    break
            curr_contrast_size = len(curr_contrasted_actions)
            if curr_contrast_size < contrast_size:
                contrast_size = curr_contrast_size
            batch_contrasted_actions.append(curr_contrasted_actions[:contrast_size])
        batch_contrasted_actions = [x[:contrast_size] for x in batch_contrasted_actions]
        return batch_contrasted_actions

    def tokenize_target(self, action_list):
        array_list = [self.sentence_to_array(x, vocabulary = 'target') for x in action_list]
        array_length = [len(x) for x in array_list]
        return array_list, array_length

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
                if len(self._examples) >= max_examples:
                    self._examples = np.array(self._examples)
                    logger.info(f"total augmentations: {self.aug_num}")    
                    return
            empty_example = {}
            input_commands = example["input_command"]
            target_commands = example["target_command"]
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue
            #equivalent_target_commands = example["equivalent_target_command"]
            input_array = self.sentence_to_array(input_commands, vocabulary="input")
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            #equivalent_target_array = self.sentence_to_array(equivalent_target_commands, vocabulary="target")
            empty_example["input_tensor"] = torch.tensor(input_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long, device=device).unsqueeze(
                dim=0)

            empty_example['input_length'] = len(input_array)
            empty_example['target_length'] = len(target_array)
            self._examples.append(empty_example)

        self._examples = np.array(self._examples)
        logger.info(f"total augmentations: {self.aug_num}")          

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

    def __getitem__(self, index):
        sample = self._examples[index]
        return sample
    
    def __len__(self):
        return len(self._examples)

    def collate_fn(self, examples):
        batch_target_actions = []
        target_lengths = []
        input_lengths = []
        for data_idx in range(len(examples)):
            curr_example = examples[data_idx]
            orig_target = curr_example['target_tensor'].squeeze(0)  ## seq_len
            orig_target_action = self.tensor_to_string(orig_target)
            batch_target_actions.append(orig_target_action)

            target_len = curr_example['target_length']
            target_lengths.append(target_len)
            input_len = curr_example['input_length']
            input_lengths.append(input_len)


        target_lengths = np.array(target_lengths)
        input_lengths = np.array(input_lengths)


        contrast_actions = self.contrast_aug_manually(batch_target_actions)  ## list of list:  batch_size    contrast_size 
        contrast_actions_from_batch = self.contrast_aug_from_batch(batch_target_actions)
        combined_contrast_actions = []
        for idx in range(len(contrast_actions)):
            combined = contrast_actions[idx] + contrast_actions_from_batch[idx]
            combined_contrast_actions.append(combined)

        batch_contrast_array = []
        batch_contrast_length = []
        for item in combined_contrast_actions:
            array_list, length_list = self.tokenize_target(item)
            batch_contrast_array.append(array_list)
            batch_contrast_length.append(length_list)
        batch_contrast_length = np.stack(batch_contrast_length, axis = 0)
        batch_max_len = np.max(batch_contrast_length, axis = 1)
        
        all_target_len = np.stack([batch_max_len, target_lengths], axis = 0)
        max_target_len = np.max(all_target_len, axis = 0)


        max_length = np.max(input_lengths + max_target_len) - 1


        # for example in examples:
        input_batch = []
        token_type_batch = []
        labels_batch = []

        for idx in range(len(examples)):
            example = examples[idx]
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

            input_list = [padded_input]
            segment_list = [token_type_tensor]
            # input_batch.append(padded_input)
            # token_type_batch.append(token_type_tensor)
            for aug_idx in range(len(batch_contrast_array[idx])):
                curr_contrast_target = batch_contrast_array[idx][aug_idx]
                curr_target_tensor = torch.tensor(curr_contrast_target, dtype=torch.long, device=device).unsqueeze(dim = 0)
                input_x = torch.cat([example["input_tensor"], curr_target_tensor[:, 1:]], dim = 1)
                sen2_len = curr_target_tensor.size(1) - 1
                token_type_list = sen1_len * [0] + sen2_len * [1]
                token_type_tensor = torch.LongTensor(token_type_list).unsqueeze(0).to(device)

                to_pad_input = max_length - input_x.size(1)
                padded_input = torch.cat([
                    input_x,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                token_type_tensor = torch.cat([
                    token_type_tensor,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                assert token_type_tensor.size(1) == padded_input.size(1), "%d -- %d "%(token_type_tensor.size(1), padded_input.size(1))
                
                input_list.append(padded_input)
                segment_list.append(token_type_tensor)        
            input_list = torch.cat(input_list, dim = 0)
            segment_list = torch.cat(segment_list, dim = 0)
            rand_order = torch.randperm(input_list.size(0), device = device)
            input_list = input_list[rand_order]
            segment_list = segment_list[rand_order]
            label = torch.argmin(rand_order)  ## find wherer 0 is
            labels_batch.append(label)
            input_batch.append(input_list)
            token_type_batch.append(segment_list)
        
        labels = torch.LongTensor(labels_batch).to(device)

        return torch.stack(input_batch, dim=0), torch.stack(token_type_batch, dim = 0), labels,


    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size


class MCGScanDataset(Dataset):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens
    for each batch, randomly sample contrast data
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, insertion = 'mid', aug_prob = 1.0,
                 white_portion = 0.5, aug_strategy = 'rand', max_white_num = 10, contrast_size = 20, 
                 contrast_from_batch_size = 10, length_control = False,
                 max_examples = None, 
                 max_len = 100, min_len = 0,
                 less_length_label = False,
                 ):
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
        self.length_control = length_control
        if length_control:
            self.contrast_creator = ContrastCreatorControl(self.target_vocabulary, white_token = self.white_token, use_white = False)
        else:
            self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token, use_white = False)


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
        self.contrast_size = contrast_size
        self.contrast_from_batch_size = contrast_from_batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.less_length_label = less_length_label
        self.read_dataset(max_examples = max_examples)

    def length_map(self, length):
        if self.length_control:
            if length <= 3:
                return 0
            elif length <= 6:
                return 1
            elif length <= 9:
                return 2
            elif length <= 12:
                return 3
            elif length <= 15:
                return 4
            else:
                print(length)
                raise NotImplementedError
        else:
            if length <= 4:
                return 0
            elif length <= 8:
                return 1
            elif length <= 12:
                return 2
            elif length <= 16:
                return 3
            elif length <= 20:
                return 4
            else:
                raise NotImplementedError

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

    # def tensor_to_string(self, batch_tensor):
    #     action_list = []
    #     for i in range(batch_tensor.size(0)):
    #         action = self.array_to_sentence(batch_tensor[i][1:-1], vocabulary = 'target')
    #         action_list.append(action)
    #     return action_list

    def tensor_to_string(self, tensor):
        action = self.array_to_sentence(tensor[1:-1], vocabulary = 'target')
        return action

    def contrast_aug_manually(self, batch_actions):
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        for i in range(batch_size):
            contrasted_actions = []
            orig_action = batch_actions[i]
            for j in range(self.contrast_size):
                contra_action = self.contrast_creator.contrast_neg_num1(orig_command = orig_action)
                contrasted_actions.append(contra_action)
            batch_contrasted_actions.append(contrasted_actions)
        return batch_contrasted_actions



    def contrast_aug_from_batch(self, batch_actions):
        contrast_size = self.contrast_from_batch_size
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        copy_batch_actions = copy.deepcopy(batch_actions)
        for i in range(batch_size):
            curr_contrasted_actions = []
            random.shuffle(copy_batch_actions)
            for j in range(batch_size):
                if batch_actions[i] != copy_batch_actions[j]:
                    curr_contrasted_actions.append(copy_batch_actions[j])
                if len(curr_contrasted_actions) >= contrast_size:
                    break
            curr_contrast_size = len(curr_contrasted_actions)
            if curr_contrast_size < contrast_size:
                contrast_size = curr_contrast_size
            batch_contrasted_actions.append(curr_contrasted_actions[:contrast_size])
        batch_contrasted_actions = [x[:contrast_size] for x in batch_contrasted_actions]
        return batch_contrasted_actions

    def tokenize_target(self, action_list):
        array_list = [self.sentence_to_array(x, vocabulary = 'target') for x in action_list]
        array_length = [len(x) for x in array_list]
        return array_list, array_length


    def read_dataset(self, max_examples=None, simple_situation_representation=True) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        """
        logger.info("Converting dataset to tensors...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split, simple_situation_representation, max_examples = max_examples, max_len = self.max_len, min_len = self.min_len)):
            if max_examples:
                if len(self._examples) > max_examples:
                    self._examples = np.array(self._examples)
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
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue
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
            empty_example['input_length'] = len(input_commands)
            empty_example['target_length'] = len(target_commands)

            self._examples.append(empty_example)

        self._examples = np.array(self._examples)
        logger.info(f"total augmentations: {self.aug_num}")          

    def __getitem__(self, index):
        return self._examples[index]
    
    def __len__(self):
        return len(self._examples)

    def collate_fn(self, examples):
        input_lengths = np.array([x['input_length'] for x in examples]) + 2
        target_lengths = np.array([x['target_length'] for x in examples]) + 2

        batch_target_actions = []
        for data_idx in range(len(examples)):
            curr_example = examples[data_idx]
            orig_target = curr_example['target_tensor'].squeeze(0)  ## seq_len
            orig_target_action = self.tensor_to_string(orig_target)
            batch_target_actions.append(orig_target_action)
        # np.random.seed(111)
        # random.seed(111)
        # torch.cuda.manual_seed(111)
        contrast_actions = self.contrast_aug_manually(batch_target_actions)  ## list of list:  batch_size    contrast_size 
        contrast_actions_from_batch = self.contrast_aug_from_batch(batch_target_actions)
        combined_contrast_actions = []
        for idx in range(len(contrast_actions)):
            combined = contrast_actions[idx] + contrast_actions_from_batch[idx]
            # combined = [batch_target_actions[idx]] + contrast_actions[idx] + contrast_actions_from_batch[idx]

            combined_contrast_actions.append(combined)

        batch_contrast_array = []
        batch_contrast_length = []
        for item in combined_contrast_actions:
            array_list, length_list = self.tokenize_target(item)
            batch_contrast_array.append(array_list)
            batch_contrast_length.append(length_list)
        batch_contrast_length = np.stack(batch_contrast_length, axis = 0)
        batch_max_len = np.max(batch_contrast_length, axis = 1)
        
        all_target_len = np.stack([batch_max_len, target_lengths], axis = 0)
        max_target_len = np.max(all_target_len, axis = 0)


        max_length = np.max(input_lengths + max_target_len) - 1


        # for example in examples:
        input_batch = []
        token_type_batch = []
        situation_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        agent_positions_batch = []
        target_positions_batch = []
        labels_batch = []

        choice_len_batch = []
        for idx in range(len(examples)):
            example = examples[idx]
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
            situation_batch.append(example["situation_tensor"])
            situation_representation_batch.append(example["situation_representation"])
            derivation_representation_batch.append(example["derivation_representation"])
            agent_positions_batch.append(example["agent_position"])
            target_positions_batch.append(example["target_position"])

            input_list = [padded_input]
            segment_list = [token_type_tensor]
            if self.less_length_label:
                choice_len_list = [self.length_map(sen2_len - 1)]
            else:
                choice_len_list = [sen2_len - 2]
            for aug_idx in range(len(batch_contrast_array[idx])):
                curr_contrast_target = batch_contrast_array[idx][aug_idx]
                curr_target_tensor = torch.tensor(curr_contrast_target, dtype=torch.long, device=device).unsqueeze(dim = 0)
                input_x = torch.cat([example["input_tensor"], curr_target_tensor[:, 1:]], dim = 1)
                sen2_len = curr_target_tensor.size(1) - 1
                token_type_list = sen1_len * [0] + sen2_len * [1]
                token_type_tensor = torch.LongTensor(token_type_list).unsqueeze(0).to(device)

                to_pad_input = max_length - input_x.size(1)
                padded_input = torch.cat([
                    input_x,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                token_type_tensor = torch.cat([
                    token_type_tensor,
                    torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                assert token_type_tensor.size(1) == padded_input.size(1), "%d -- %d "%(token_type_tensor.size(1), padded_input.size(1))
                
                input_list.append(padded_input)
                segment_list.append(token_type_tensor)   
                if self.less_length_label:
                    choice_len_list.append(self.length_map(sen2_len - 1))     
                else:
                    choice_len_list.append(sen2_len - 2)
            input_list = torch.cat(input_list, dim = 0)
            segment_list = torch.cat(segment_list, dim = 0)
            choice_len_list = torch.tensor(choice_len_list, device = device).long()
            
            # rand_order = torch.randperm(input_list.size(0), device = device)
            # input_list = input_list[rand_order]
            # segment_list = segment_list[rand_order]
            # choice_len_list = choice_len_list[rand_order]
            # input_list = input_list.index_select(0, rand_order)
            # segment_list = segment_list.index_select(0, rand_order)
            # choice_len_list = choice_len_list.index_select(0, rand_order)

            # label = torch.argmin(rand_order)  ## find wherer 0 is
            label = 0
            labels_batch.append(label)
            input_batch.append(input_list)
            token_type_batch.append(segment_list)
            choice_len_batch.append(choice_len_list)
            
        labels = torch.LongTensor(labels_batch).to(device)
        choice_len_batch = torch.stack(choice_len_batch, dim = 0)
        res_dict = {
            'input_batch': torch.stack(input_batch, dim=0),
            'token_type_batch': torch.stack(token_type_batch, dim = 0),
            'input_length': input_lengths,
            'derivation_representation_batch': derivation_representation_batch,
            'situation_batch': torch.cat(situation_batch, dim=0),
            'situation_representation_batch': situation_representation_batch,
            'labels': labels,
            'choice_len_batch': choice_len_batch,
            'agent_positions_batch': torch.cat(agent_positions_batch, dim=0),
            'target_positions_batch': torch.cat(target_positions_batch, dim=0),
            "choice_list": batch_contrast_array,
        }
        return res_dict
            # return (torch.stack(input_batch, dim=0), torch.stack(token_type_batch, dim = 0), input_lengths, derivation_representation_batch,
            #        torch.cat(situation_batch, dim=0), situation_representation_batch,
            #        labels,
            #        torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

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


class MCGScanDataset_Simplefeat1(Dataset):
    """
    Loads a GroundedScan instance from a specified location.
    do not insert white space tokens to construct positive/negative tokens
    for each batch, randomly sample contrast data
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, insertion = 'mid', aug_prob = 1.0,
                 white_portion = 0.5, aug_strategy = 'rand', max_white_num = 10, contrast_size = 20, 
                 contrast_from_batch_size = 10, length_control = False,
                 max_examples = None, 
                 max_len = 100, min_len = 0,
                 ):
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
        if length_control:
            self.contrast_creator = ContrastCreatorControl(self.target_vocabulary, white_token = self.white_token, use_white = False)
        else:
            self.contrast_creator = ContrastCreator(self.target_vocabulary, white_token = self.white_token, use_white = False)

        vocab_dict = self.target_vocabulary.to_dict()
        word2idx = vocab_dict['word_to_idx']
        sos_token = vocab_dict['sos_token']
        eos_token = vocab_dict['eos_token']
        pad_token = vocab_dict['pad_token']
        white_token = self.white_token

        self.target_vocab_list = [x for x in word2idx.keys() if x not in [sos_token, eos_token, pad_token, white_token]]
        self.target_word2idx = {self.target_vocab_list[i]:i for i in range(len(self.target_vocab_list))}
        print(self.target_word2idx)

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
        self.contrast_size = contrast_size
        self.contrast_from_batch_size = contrast_from_batch_size
        self.max_len = max_len
        self.min_len = min_len
        self.read_dataset(max_examples = max_examples)

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

    # def tensor_to_string(self, batch_tensor):
    #     action_list = []
    #     for i in range(batch_tensor.size(0)):
    #         action = self.array_to_sentence(batch_tensor[i][1:-1], vocabulary = 'target')
    #         action_list.append(action)
    #     return action_list

    def tensor_to_string(self, tensor):
        action = self.array_to_sentence(tensor[1:-1], vocabulary = 'target')
        return action

    def contrast_aug_manually(self, batch_actions):
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        for i in range(batch_size):
            contrasted_actions = []
            orig_action = batch_actions[i]
            for j in range(self.contrast_size):
                contra_action = self.contrast_creator.contrast_neg_num1(orig_command = orig_action)
                contrasted_actions.append(contra_action)
            batch_contrasted_actions.append(contrasted_actions)
        return batch_contrasted_actions



    def contrast_aug_from_batch(self, batch_actions):
        contrast_size = self.contrast_from_batch_size
        batch_contrasted_actions = []
        batch_size = len(batch_actions)
        copy_batch_actions = copy.deepcopy(batch_actions)
        for i in range(batch_size):
            curr_contrasted_actions = []
            random.shuffle(copy_batch_actions)
            for j in range(batch_size):
                if batch_actions[i] != copy_batch_actions[j]:
                    curr_contrasted_actions.append(copy_batch_actions[j])
                if len(curr_contrasted_actions) >= contrast_size:
                    break
            curr_contrast_size = len(curr_contrasted_actions)
            if curr_contrast_size < contrast_size:
                contrast_size = curr_contrast_size
            batch_contrasted_actions.append(curr_contrasted_actions[:contrast_size])
        batch_contrasted_actions = [x[:contrast_size] for x in batch_contrasted_actions]
        return batch_contrasted_actions

    def tokenize_target(self, action_list):
        array_list = [self.tokenize_one(x) for x in action_list]
        return array_list

    def tokenize_one(self, action_seq):
        freq_list = np.zeros(5)
        for action in action_seq:
            freq_list[self.target_word2idx[action]] += 1
        freq_list /= len(action_seq)
        return freq_list

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

            batch_target_actions = []
            for data_idx in range(len(examples)):
                curr_example = examples[data_idx]
                orig_target = curr_example['target_tensor'].squeeze(0)  ## seq_len
                orig_target_action = self.tensor_to_string(orig_target)
                batch_target_actions.append(orig_target_action)

            contrast_actions = self.contrast_aug_manually(batch_target_actions)  ## list of list:  batch_size    contrast_size 
            contrast_actions_from_batch = self.contrast_aug_from_batch(batch_target_actions)
            combined_contrast_actions = []
            for idx in range(len(contrast_actions)):
                combined = contrast_actions[idx] + contrast_actions_from_batch[idx]
                combined_contrast_actions.append(combined)

            batch_contrast_array = []
            batch_contrast_length = []
            for item in combined_contrast_actions:
                array_list, length_list = self.tokenize_target(item)
                batch_contrast_array.append(array_list)
                batch_contrast_length.append(length_list)
            batch_contrast_length = np.stack(batch_contrast_length, axis = 0)
            batch_max_len = np.max(batch_contrast_length, axis = 1)
            
            all_target_len = np.stack([batch_max_len, target_lengths], axis = 0)
            max_target_len = np.max(all_target_len, axis = 0)


            max_length = np.max(input_lengths + max_target_len) - 1


            # for example in examples:
            input_batch = []
            token_type_batch = []
            situation_batch = []
            situation_representation_batch = []
            derivation_representation_batch = []
            agent_positions_batch = []
            target_positions_batch = []
            labels_batch = []

            for idx in range(len(examples)):
                example = examples[idx]
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
                situation_batch.append(example["situation_tensor"])
                situation_representation_batch.append(example["situation_representation"])
                derivation_representation_batch.append(example["derivation_representation"])
                agent_positions_batch.append(example["agent_position"])
                target_positions_batch.append(example["target_position"])

                input_list = [padded_input]
                segment_list = [token_type_tensor]
                # input_batch.append(padded_input)
                # token_type_batch.append(token_type_tensor)
                for aug_idx in range(len(batch_contrast_array[idx])):
                    curr_contrast_target = batch_contrast_array[idx][aug_idx]
                    curr_target_tensor = torch.tensor(curr_contrast_target, dtype=torch.long, device=device).unsqueeze(dim = 0)
                    input_x = torch.cat([example["input_tensor"], curr_target_tensor[:, 1:]], dim = 1)
                    sen2_len = curr_target_tensor.size(1) - 1
                    token_type_list = sen1_len * [0] + sen2_len * [1]
                    token_type_tensor = torch.LongTensor(token_type_list).unsqueeze(0).to(device)

                    to_pad_input = max_length - input_x.size(1)
                    padded_input = torch.cat([
                        input_x,
                        torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                    token_type_tensor = torch.cat([
                        token_type_tensor,
                        torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                    assert token_type_tensor.size(1) == padded_input.size(1), "%d -- %d "%(token_type_tensor.size(1), padded_input.size(1))
                    
                    input_list.append(padded_input)
                    # situation_batch.append(example["situation_tensor"])
                    # situation_representation_batch.append(example["situation_representation"])
                    # derivation_representation_batch.append(example["derivation_representation"])
                    # agent_positions_batch.append(example["agent_position"])
                    # target_positions_batch.append(example["target_position"])
                    segment_list.append(token_type_tensor)        
                input_list = torch.cat(input_list, dim = 0)
                segment_list = torch.cat(segment_list, dim = 0)
                rand_order = torch.randperm(input_list.size(0), device = device)
                input_list = input_list[rand_order]
                segment_list = segment_list[rand_order]
                label = torch.argmin(rand_order)  ## find wherer 0 is
                labels_batch.append(label)
                input_batch.append(input_list)
                token_type_batch.append(segment_list)
            
            labels = torch.LongTensor(labels_batch).to(device)

            yield (torch.stack(input_batch, dim=0), torch.stack(token_type_batch, dim = 0), input_lengths, derivation_representation_batch,
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
                    self._examples = np.array(self._examples)
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
            if len(target_commands) > self.max_len or len(target_commands) < self.min_len:
                continue
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
            empty_example['input_length'] = len(input_commands)
            empty_example['target_length'] = len(target_commands)

            self._examples.append(empty_example)

        self._examples = np.array(self._examples)
        logger.info(f"total augmentations: {self.aug_num}")          

    def __getitem__(self, index):
        return self._examples[index]
    
    def __len__(self):
        return len(self._examples)

    def collate_fn(self, examples):
        input_lengths = np.array([x['input_length'] for x in examples]) + 2

        batch_target_actions = []
        for data_idx in range(len(examples)):
            curr_example = examples[data_idx]
            orig_target = curr_example['target_tensor'].squeeze(0)  ## seq_len
            orig_target_action = self.tensor_to_string(orig_target)
            batch_target_actions.append(orig_target_action)

        contrast_actions = self.contrast_aug_manually(batch_target_actions)  ## list of list:  batch_size    contrast_size 
        contrast_actions_from_batch = self.contrast_aug_from_batch(batch_target_actions)
        combined_contrast_actions = []
        for idx in range(len(contrast_actions)):
            combined = [batch_target_actions[idx]] + contrast_actions[idx] + contrast_actions_from_batch[idx]
            combined_contrast_actions.append(combined)

        batch_contrast_array = []
        batch_contrast_length = []
        for item in combined_contrast_actions:
            array_list = self.tokenize_target(item)
            batch_contrast_array.append(array_list)
        
        max_length = np.max(input_lengths, axis = 0)


        # for example in examples:
        input_batch = []
        token_type_batch = []
        situation_batch = []
        situation_representation_batch = []
        derivation_representation_batch = []
        agent_positions_batch = []
        target_positions_batch = []
        labels_batch = []

        choice_batch = []
        for idx in range(len(examples)):
            example = examples[idx]
            input_x = example["input_tensor"]
            sen1_len = example["input_tensor"].size(1)

            to_pad_input = max_length - input_x.size(1)
            padded_input = torch.cat([
                input_x,
                torch.zeros(int(to_pad_input), dtype=torch.long, device=device).unsqueeze(0)], dim=1)

            input_batch.append(padded_input)
            situation_batch.append(example["situation_tensor"])
            situation_representation_batch.append(example["situation_representation"])
            derivation_representation_batch.append(example["derivation_representation"])
            agent_positions_batch.append(example["agent_position"])
            target_positions_batch.append(example["target_position"])

            target_list = torch.tensor(batch_contrast_array[idx], dtype=torch.float, device=device)
            
            rand_order = torch.randperm(target_list.size(0), device = device)
            target_list = target_list[rand_order]

            label = torch.argmin(rand_order)  ## find wherer 0 is
            labels_batch.append(label)
            choice_batch.append(target_list)
            
        labels = torch.LongTensor(labels_batch).to(device)
        choice_batch = torch.stack(choice_batch, dim = 0)
        res_dict = {
            'input_batch': torch.cat(input_batch, dim=0),
            'input_length': input_lengths,
            'derivation_representation_batch': derivation_representation_batch,
            'situation_batch': torch.cat(situation_batch, dim=0),
            'situation_representation_batch': situation_representation_batch,
            'labels': labels,
            'choice_batch': choice_batch,
            'agent_positions_batch': torch.cat(agent_positions_batch, dim=0),
            'target_positions_batch': torch.cat(target_positions_batch, dim=0)
        }
        return res_dict

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



