import argparse
import logging
import os
import torch
from seq2seq.evaluate import evaluate_disc

from seq2seq.gSCAN_dataset import GScanNLI
from seq2seq.my_model import DiscTextOnly
from seq2seq.helpers import log_parameters
from seq2seq.tri_state_lr_scheduler import GradualWarmupScheduler

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm

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
parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
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
parser.add_argument("--model_type", type=str, default='cls', choices=['cls', 'mc'],)
parser.add_argument("--inf_type", type=str, default='orig', choices=['orig', 'no_eos', 'len_reg','eos_distribution', 'beam_search'],)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--aug_prob", type=float, default=0.7)
parser.add_argument("--white_portion", type=float, default=0.5)
parser.add_argument("--aug_strategy", type=str, default='rand', choices=['rand', 'fixed'],)
parser.add_argument("--insertion", type=str, default='mid', choices=['start', 'mid'],)
parser.add_argument("--max_white_num", type=int, default=5)
parser.add_argument("--num_transformer_layers", type=int, default=1)
parser.add_argument('--warmup', type=float, default=2000)
parser.add_argument("--warmup_lr", type=float, default=1e-6)
parser.add_argument("--data_type", type=str, default='v1', choices=['v1', 'v2', 'v3', 'v4', 'mc'],) ## v1: original   v2: not white space  v3: no white space, only delete
parser.add_argument("--contrast_size", type=int, default=10)
parser.add_argument("--contrast_from_batch_size", type=int, default=9)
parser.add_argument("--length_control", action = 'store_true')

def main(flags):
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))


    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        train_dataset = GScanNLI(data_path, flags["data_directory"], split = 'train', input_vocabulary_file=flags["input_vocab_path"],
                                  target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                  length_control = flags['length_control'], 
                                  contrast_size= flags['contrast_size'], 
                                  contrast_from_batch_size = flags['contrast_from_batch_size'],
                                  max_examples = flags['max_training_examples']
                                  )
        valid_dataset = GScanNLI(data_path, flags["data_directory"], split = 'dev', input_vocabulary_file=flags["input_vocab_path"],
                                  target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                  length_control = flags['length_control'], 
                                  contrast_size= flags['contrast_size'], 
                                  contrast_from_batch_size = flags['contrast_from_batch_size'],
                                  max_examples = flags['max_testing_examples'],
                                  )
        train_loader = DataLoader(train_dataset, batch_size = flags['training_batch_size'], shuffle = True, collate_fn = train_dataset.collate_fn,)
        valid_loader = DataLoader(valid_dataset, batch_size = flags['test_batch_size'], shuffle = False, collate_fn = valid_dataset.collate_fn,)

        logger.info("Done Loading Training set.")
        logger.info("  Loaded {} training examples.".format(train_dataset.num_examples))
        logger.info("  Input vocabulary size training set: {}".format(train_dataset.input_vocabulary_size))
        logger.info("  Most common input words: {}".format(train_dataset.input_vocabulary.most_common(5)))
        logger.info("  Output vocabulary size training set: {}".format(train_dataset.target_vocabulary_size))
        logger.info("  Most common target words: {}".format(train_dataset.target_vocabulary.most_common(5)))



        model = DiscTextOnly(input_vocabulary_size=train_dataset.input_vocabulary_size,
                            target_vocabulary_size=train_dataset.target_vocabulary_size,
                            input_padding_idx=train_dataset.input_vocabulary.pad_idx,
                            target_pad_idx=train_dataset.target_vocabulary.pad_idx,
                            target_eos_idx=train_dataset.target_vocabulary.eos_idx,
                            **flags)

        lr_decay = flags['lr_decay']
        max_training_iterations = flags['max_training_iterations']
        print_every = flags['print_every']
        evaluate_every = flags['evaluate_every']
        adam_beta_1 = flags['adam_beta_1']
        adam_beta_2 = flags['adam_beta_2']
        lr_decay_steps = flags['lr_decay_steps']
        warmup = flags['warmup']
        test_batch_size = flags['test_batch_size']
        learning_rate = flags['learning_rate']



        model = model.cuda() if use_cuda else model
        log_parameters(model)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
        after_scheduler = LambdaLR(optimizer,
                            lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))
        
        scheduler = GradualWarmupScheduler(optimizer, 1.0, total_epoch = warmup, after_scheduler = after_scheduler)

        start_iteration = 1
        best_iteration = 1
        best_accuracy = 0
        best_exact_match = 0
        best_loss = float('inf')


        if flags['resume_from_file']:
            assert os.path.isfile(flags['resume_from_file']), "No checkpoint found at {}".format(flags['resume_from_file'])
            logger.info("Loading checkpoint from file at '{}'".format(flags['resume_from_file']))
            optimizer_state_dict = model.load_model(flags['resume_from_file'])
            optimizer.load_state_dict(optimizer_state_dict)
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags['resume_from_file'], start_iteration))
            for i in range(start_iteration):
                optimizer.step()
                scheduler.step()



        logger.info("Training starts..")
        training_iteration = start_iteration
        while training_iteration < max_training_iterations:
            for (input_batch, token_type_ids, labels) in train_loader:
                is_best = False
                model.train()

                # Forward pass.
                logits = model(commands_input=input_batch, token_type_ids = token_type_ids)
                loss = model.get_loss(logits, labels)

                # Backward pass and update model parameters.
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.update_state(is_best=is_best)

                # Print current metrics.
                if training_iteration % print_every == 0:
                    loss, acc, recall, precision, F1 = model.get_metrics(logits, labels)
                    learning_rate = scheduler.get_lr()[0]
                    logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f, learning_rate %.5f"% (training_iteration, loss, acc,
                                                                    learning_rate, ))

                # Evaluate on test set.
                if training_iteration % evaluate_every == 0:
                    with torch.no_grad():
                        model.eval()
                        logger.info("Evaluating..")
                        
                        corr_total = 0
                        total = 0
                        all_pred = []
                        all_orig = []
                        for (input_batch, token_type_ids, labels) in valid_loader:
                            logits = model(commands_input=input_batch, token_type_ids = token_type_ids)
                            pred_labels = torch.argmax(logits, dim = 1)
                            corr = (pred_labels == labels).sum().item()
                            corr_total += corr
                            total += input_batch.size(0)
                            all_pred.append(pred_labels)
                            all_orig.append(labels)
                        all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
                        all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
                        precision = precision_score(all_orig, all_pred, average = 'macro')
                        recall = recall_score(all_orig, all_pred, average = 'macro')
                        F1 = f1_score(all_orig, all_pred, average = 'macro')
                        accuracy = corr_total / total
                        

                        logger.info("  Evaluation Accuracy: %5.3f, recall: %.3f, precision: %.3f, F1: %.3f" % (acc, recall, precision, F1))
                        if acc > best_exact_match:
                            is_best = True
                            best_accuracy = acc
                            best_recall = recall
                            best_precision = precision
                            best_F1 = F1
                            model.update_state(acc=best_accuracy, F1 = best_F1, recall = best_recall, precision = best_precision, is_best=is_best)
                        file_name = "checkpoint.pth.tar".format(str(training_iteration))
                        if is_best:
                            model.save_checkpoint(file_name=file_name, is_best=is_best,
                                                optimizer_state_dict=optimizer.state_dict())

                training_iteration += 1
                if training_iteration > max_training_iterations:
                    break
        logger.info("Finished training.")


    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))


        test_dataset = GScanNLI(data_path, flags["data_directory"], split = 'dev', input_vocabulary_file=flags["input_vocab_path"],
                                  target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                  length_control = flags['length_control'], 
                                  contrast_size= flags['contrast_size'], 
                                  contrast_from_batch_size = flags['contrast_from_batch_size'],
                                  max_examples = flags['max_testing_examples'],
                                  )
        test_loader = DataLoader(test_dataset, batch_size = flags['test_batch_size'], shuffle = False, collate_fn = test_dataset.collate_fn,)

        logger.info("Done Loading {} dataset split.".format(flags["split"]))
        logger.info("  Loaded {} examples.".format(test_dataset.num_examples))
        logger.info("  Input vocabulary size: {}".format(test_dataset.input_vocabulary_size))
        logger.info("  Most common input words: {}".format(test_dataset.input_vocabulary.most_common(5)))
        logger.info("  Output vocabulary size: {}".format(test_dataset.target_vocabulary_size))
        logger.info("  Most common target words: {}".format(test_dataset.target_vocabulary.most_common(5)))
        
        model = DiscTextOnly(input_vocabulary_size=train_dataset.input_vocabulary_size,
                            target_vocabulary_size=train_dataset.target_vocabulary_size,
                            input_padding_idx=train_dataset.input_vocabulary.pad_idx,
                            target_pad_idx=train_dataset.target_vocabulary.pad_idx,
                            target_eos_idx=train_dataset.target_vocabulary.eos_idx,
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
        with torch.no_grad():
            model.eval()
            logger.info("Evaluating..")
            
            corr_total = 0
            total = 0
            all_pred = []
            all_orig = []
            for (input_batch, token_type_ids, labels) in test_loader:
                logits = model(commands_input=input_batch, token_type_ids = token_type_ids)
                pred_labels = torch.argmax(logits, dim = 1)
                corr = (pred_labels == labels).sum().item()
                corr_total += corr
                total += input_batch.size(0)
                all_pred.append(pred_labels)
                all_orig.append(labels)
            all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
            all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
            precision = precision_score(all_orig, all_pred, average = 'macro')
            recall = recall_score(all_orig, all_pred, average = 'macro')
            F1 = f1_score(all_orig, all_pred, average = 'macro')
            accuracy = corr_total / total
            
            logger.info("  Evaluation Accuracy: %5.3f, recall: %.3f, precision: %.3f, F1: %.3f" % (acc, recall, precision, F1))

if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
