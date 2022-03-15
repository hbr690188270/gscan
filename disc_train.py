import argparse
import logging
import os
import torch
from seq2seq.evaluate import evaluate_disc

from seq2seq.gSCAN_dataset import GroundedScanDataset, MultipleChoiceGScan, GroundedScanDatasetContrastV2, GroundedScanDatasetContrastV3, GroundedScanDatasetContrastV4
from seq2seq.my_model import Discriminator, DiscriminatorMultipleChoice
from seq2seq.train import disc_train
from seq2seq.predict import (predict_and_save, predict_and_save_beamsearch, predict_and_save_disc, predict_and_save_noeos, predict_and_save_pad, predict_and_save_transformer, 
                                predict_and_save_noeos_transformer, predict_and_save_lenreg, predict_and_save_eos_distribution)

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False

if use_cuda:
    logger.info("Using CUDA.")
    logger.info("Cuda version: {}".format(torch.version.cuda))

parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

# General arguments
parser.add_argument("--mode", type=str, default="run_tests", help="train, test or predict", required=True)
parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                           "saved. Will be created if doesn't exist.")
parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")

# Data arguments
parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
parser.add_argument("--data_directory", type=str, default="data/uniform_dataset", help="Path to folder with data.")
parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                    help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                    help="Whether to generate vocabularies based on the data.")
parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                    help="Whether to use previously saved vocabularies.")

# Training and learning arguments
parser.add_argument("--training_batch_size", type=int, default=50)
parser.add_argument("--k", type=int, default=0, help="How many examples from the adverb_1 split to move to train.")
parser.add_argument("--test_batch_size", type=int, default=12, help="Currently only 1 supported due to decoder.")
parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_decay_steps', type=float, default=20000)
parser.add_argument("--adam_beta_1", type=float, default=0.9)
parser.add_argument("--adam_beta_2", type=float, default=0.999)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                     "test set (without teacher forcing).")
parser.add_argument("--max_training_iterations", type=int, default=100000)

# Testing and predicting arguments
parser.add_argument("--max_testing_examples", type=int, default=None)
parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
                                                                       "is stopped regardless of whether an EOS token "
                                                                       "was generated.")
parser.add_argument("--output_file_name", type=str, default="predict.json")

# Situation Encoder arguments
parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                    action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                              "For more information, read grounded SCAN documentation.")
parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                    action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                               "For more information, read grounded SCAN documentation.")
parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                   "model.")
parser.add_argument("--cnn_dropout_p", type=float, default=0.1, help="Dropout applied to the output features of the "
                                                                     "world state model.")

# Command Encoder arguments
parser.add_argument("--encoder_hidden_size", type=int, default=128)
parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")

# Other arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_type", type=str, default='cls', choices=['cls', 'mc','mc_v2'],)
parser.add_argument("--inf_type", type=str, default='orig', choices=['orig', 'no_eos', 'len_reg','eos_distribution', 'beam_search'],)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--aug_prob", type=float, default=0.7)
parser.add_argument("--white_portion", type=float, default=0.5)
parser.add_argument("--aug_strategy", type=str, default='rand', choices=['rand', 'fixed'],)
parser.add_argument("--insertion", type=str, default='mid', choices=['start', 'mid'],)
parser.add_argument("--max_white_num", type=int, default=5)
parser.add_argument("--num_transformer_layers", type=int, default=1)
parser.add_argument('--warmup', type=float, default=20000)
parser.add_argument("--warmup_lr", type=float, default=1e-6)
parser.add_argument("--data_type", type=str, default='mc', choices=['v1', 'v2', 'v3', 'v4', 'mc', 'simple1'],) ## v1: original   v2: not white space  v3: no white space, only delete
parser.add_argument("--contrast_size", type=int, default=20)
parser.add_argument("--contrast_from_batch_size", type=int, default=20)
parser.add_argument("--length_control", action = 'store_true')

def main(flags):
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))

    if not flags["simple_situation_representation"]:
        raise NotImplementedError("Full RGB input image not implemented. Implement or set "
                                  "--simple_situation_representation")
    # Some checks on the flags
    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    # if flags["test_batch_size"] > 1:
    #     raise NotImplementedError("Test batch size larger than 1 not implemented.")

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        disc_train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))
            if flags['data_type'] == 'v2':
                test_set = GroundedScanDatasetContrastV2(data_path, flags["data_directory"], split=split,
                                            input_vocabulary_file=flags["input_vocab_path"],
                                            target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                            k=flags["k"], 
                                            aug_prob = flags['aug_prob'], white_portion = flags['white_portion'],
                                            max_white_num = flags['max_white_num'], insertion = flags['insertion'], aug_strategy = flags['aug_strategy'],
                                            )
            elif flags['data_type'] == 'v3':
                test_set = GroundedScanDatasetContrastV3(data_path, flags["data_directory"], split=split,
                                            input_vocabulary_file=flags["input_vocab_path"],
                                            target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                            k=flags["k"], 
                                            aug_prob = flags['aug_prob'], white_portion = flags['white_portion'],
                                            max_white_num = flags['max_white_num'], insertion = flags['insertion'], aug_strategy = flags['aug_strategy'],
                                            )
            elif flags['data_type'] == 'v4':
                test_set = GroundedScanDatasetContrastV4(data_path, flags["data_directory"], split=split,
                                            input_vocabulary_file=flags["input_vocab_path"],
                                            target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                            k=flags["k"], 
                                            aug_prob = flags['aug_prob'], white_portion = flags['white_portion'],
                                            max_white_num = flags['max_white_num'], insertion = flags['insertion'], aug_strategy = flags['aug_strategy'],
                                            )
            elif flags['data_type'] == 'mc':
                test_set = MultipleChoiceGScan(data_path, flags["data_directory"], split=split,
                                                input_vocabulary_file=flags["input_vocab_path"],
                                                target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                                k=flags["k"], 
                                                aug_prob = flags['aug_prob'], white_portion = flags['white_portion'],
                                                max_white_num = flags['max_white_num'], insertion = flags['insertion'], aug_strategy = flags['aug_strategy'],
                                                contrast_size= flags['contrast_size'], contrast_from_batch_size = flags['contrast_from_batch_size'],
                                                length_control = flags['length_control'],
                                                max_len = 20,
                                                min_len = 0,
                                                )


            test_set.read_dataset(max_examples = flags['max_testing_examples'],
                                  simple_situation_representation=flags["simple_situation_representation"])
            logger.info("Done Loading {} dataset split.".format(flags["split"]))
            logger.info("  Loaded {} examples.".format(test_set.num_examples))
            logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
            logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
            logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
            logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

            if flags['model_type'] == 'cls':
                model = Discriminator(input_vocabulary_size=test_set.input_vocabulary_size,
                            target_vocabulary_size=test_set.target_vocabulary_size,
                            num_cnn_channels=test_set.image_channels,
                            input_padding_idx=test_set.input_vocabulary.pad_idx,
                            target_pad_idx=test_set.target_vocabulary.pad_idx,
                            target_eos_idx=test_set.target_vocabulary.eos_idx,
                            **flags)

            elif flags['model_type'] == 'mc':
                model = DiscriminatorMultipleChoice(input_vocabulary_size=test_set.input_vocabulary_size,
                                    target_vocabulary_size=test_set.target_vocabulary_size,
                                    num_cnn_channels=test_set.image_channels,
                                    input_padding_idx=test_set.input_vocabulary.pad_idx,
                                    target_pad_idx=test_set.target_vocabulary.pad_idx,
                                    target_eos_idx=test_set.target_vocabulary.eos_idx,
                                    **flags)


            model = model.cuda() if use_cuda else model

            # Load model and vocabularies if resuming.
            assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
            logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
            model.load_model(flags["resume_from_file"])
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
            output_file_name = "_".join([split, flags["output_file_name"]])
            output_file_path = os.path.join(flags["output_directory"], output_file_name)

            if flags['model_type'] == 'mc':
                accuracy, recall, precision, F1 = evaluate_disc(data_iterator = test_set.get_data_iterator(batch_size=flags['test_batch_size']), model=model, multi_class = True)
            else:
                accuracy, recall, precision, F1 = evaluate_disc(data_iterator = test_set.get_data_iterator(batch_size=flags['test_batch_size']), model=model,)
            logger.info("Evaluation Accuracy: %5.3f, recall: %.3f, precision: %.3f, F1: %.3f" % (accuracy, recall, precision, F1))
            
            
            # output_file = predict_and_save_disc(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
            
            # logger.info("Saved predictions to {}".format(output_file))
    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
