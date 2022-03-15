import argparse
from json import encoder
import logging
import os
import torch
from seq2seq.evaluate import evaluate_disc

from seq2seq.gSCAN_dataset import MCGScanDataset
from seq2seq.my_model import DiscOfDiscCls, DiscriminatorMC_V2, DiscriminatorMC_V3, DiscriminatorMCAdv, DiscOfDisc, DiscOfDiscCls_MLP
from seq2seq.helpers import log_parameters, entropy
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
parser.add_argument("--evaluate_every", type=int, default=2000, help="How often to evaluate the model by decoding the "
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
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout on instruction embeddings and LSTM.")

# Other arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_type", type=str, default='cls', choices=['cls', 'mc','mc_v2', 'mc_v3'],)
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
parser.add_argument("--data_type", type=str, default='mc', choices=['v1', 'v2', 'v3', 'v4', 'mc'],) ## v1: original   v2: not white space  v3: no white space, only delete
parser.add_argument("--contrast_size", type=int, default=10)
parser.add_argument("--contrast_from_batch_size", type=int, default=9)
parser.add_argument("--length_control", action = 'store_true')
parser.add_argument("--adv_beta", type=float, default=0.1)
parser.add_argument("--loss_type", type=str, default='cls', choices=['cls', 'reg'],) 
parser.add_argument("--adv_network", type=str, default='linear', choices=['linear', 'mlp'],) 
parser.add_argument("--less_length_label", action = 'store_true')
parser.add_argument("--training_type", type=str, default='mc_only', choices=['mc_only', 'gan', 'adv_only'],) 
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--maximize_tgt", type=str, default='entropy', choices=['entropy', 'disc_loss'],) 

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
        train_dataset = MCGScanDataset(data_path, flags["data_directory"], split = 'train', input_vocabulary_file=flags["input_vocab_path"],
                                        target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                        length_control = flags['length_control'], 
                                        contrast_size= flags['contrast_size'], 
                                        contrast_from_batch_size = flags['contrast_from_batch_size'],
                                        max_examples = flags['max_training_examples'],
                                        less_length_label = flags['less_length_label']
                                        )

        eval_train_dataset = MCGScanDataset(data_path, flags["data_directory"], split = 'train', input_vocabulary_file=flags["input_vocab_path"],
                                        target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                        length_control = flags['length_control'], 
                                        contrast_size= flags['contrast_size'], 
                                        contrast_from_batch_size = flags['contrast_from_batch_size'],
                                        max_examples = 10000,
                                        less_length_label = flags['less_length_label']
                                        )


        valid_dataset = MCGScanDataset(data_path, flags["data_directory"], split = 'dev', input_vocabulary_file=flags["input_vocab_path"],
                                        target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                        length_control = flags['length_control'], 
                                        contrast_size= flags['contrast_size'], 
                                        contrast_from_batch_size = flags['contrast_from_batch_size'],
                                        max_examples = flags['max_training_examples'],
                                        less_length_label = flags['less_length_label']
                                        )

        train_loader = DataLoader(train_dataset, batch_size = flags['training_batch_size'], shuffle = True, collate_fn = train_dataset.collate_fn, )
        eval_train_loader = DataLoader(eval_train_dataset, batch_size = flags['training_batch_size'], shuffle = False, collate_fn = train_dataset.collate_fn, )
        valid_loader = DataLoader(valid_dataset, batch_size = flags['test_batch_size'], shuffle = False, collate_fn = valid_dataset.collate_fn,)

        logger.info("Done Loading Training set.")
        logger.info("  Loaded {} training examples.".format(train_dataset.num_examples))
        logger.info("  Input vocabulary size training set: {}".format(train_dataset.input_vocabulary_size))
        logger.info("  Most common input words: {}".format(train_dataset.input_vocabulary.most_common(5)))
        logger.info("  Output vocabulary size training set: {}".format(train_dataset.target_vocabulary_size))
        logger.info("  Most common target words: {}".format(train_dataset.target_vocabulary.most_common(5)))

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
        training_type = flags['training_type']
        adv_beta = flags['adv_beta']
        loss_type = flags['loss_type']
        model_type = flags['model_type']
        num_gpus = flags['num_gpus']
        maximize_tgt = flags['maximize_tgt']
        adv_network = flags['adv_network']


        if model_type == 'mc':
            model = DiscriminatorMCAdv(input_vocabulary_size=train_dataset.input_vocabulary_size,
                                        target_vocabulary_size=train_dataset.target_vocabulary_size,
                                        num_cnn_channels=train_dataset.image_channels,
                                        input_padding_idx=train_dataset.input_vocabulary.pad_idx,
                                        target_pad_idx=train_dataset.target_vocabulary.pad_idx,
                                        target_eos_idx=train_dataset.target_vocabulary.eos_idx,
                                        **flags)
        elif model_type == 'mc_v2':
            model = DiscriminatorMC_V2(input_vocabulary_size=train_dataset.input_vocabulary_size,
                                        target_vocabulary_size=train_dataset.target_vocabulary_size,
                                        num_cnn_channels=train_dataset.image_channels,
                                        input_padding_idx=train_dataset.input_vocabulary.pad_idx,
                                        target_pad_idx=train_dataset.target_vocabulary.pad_idx,
                                        target_eos_idx=train_dataset.target_vocabulary.eos_idx,
                                        **flags)            
        elif model_type == 'mc_v3':
            model = DiscriminatorMC_V3(input_vocabulary_size=train_dataset.input_vocabulary_size,
                                        target_vocabulary_size=train_dataset.target_vocabulary_size,
                                        num_cnn_channels=train_dataset.image_channels,
                                        input_padding_idx=train_dataset.input_vocabulary.pad_idx,
                                        target_pad_idx=train_dataset.target_vocabulary.pad_idx,
                                        target_eos_idx=train_dataset.target_vocabulary.eos_idx,
                                        **flags)  
        
        if flags['loss_type'] == 'reg':
            length_predictor = DiscOfDisc(encoder_hidden_size = flags['encoder_hidden_size'])
        elif flags['loss_type'] == 'cls':
            if flags['less_length_label']:
                length_predictor = DiscOfDiscCls(encoder_hidden_size = 2 * flags['encoder_hidden_size'], class_num = 5)
            else:
                if adv_network == 'linear': 
                    length_predictor = DiscOfDiscCls(encoder_hidden_size = 2 * flags['encoder_hidden_size'], class_num = 15)
                elif adv_network == 'mlp':
                    length_predictor = DiscOfDiscCls_MLP(encoder_hidden_size = 2 * flags['encoder_hidden_size'], class_num = 15)

        model = model.cuda() if use_cuda else model
        length_predictor = length_predictor.cuda() if use_cuda else length_predictor
        log_parameters(model)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        length_predictor_params = [parameter for parameter in length_predictor.parameters() if parameter.requires_grad]
        optimizer_gen = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
        optimizer_disc = torch.optim.Adam(length_predictor_params, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))

        after_scheduler_gen = LambdaLR(optimizer_gen,
                            lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

        after_scheduler_disc = LambdaLR(optimizer_disc,
                            lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))


        scheduler_gen = GradualWarmupScheduler(optimizer_gen, 1.0, total_epoch = warmup, after_scheduler = after_scheduler_gen)
        scheduler_disc = GradualWarmupScheduler(optimizer_disc, 1.0, total_epoch = warmup, after_scheduler = after_scheduler_disc)

        if flags['resume_from_file']:
            assert os.path.isfile(flags['resume_from_file']), "No checkpoint found at {}".format(flags['resume_from_file'])
            logger.info("Loading checkpoint from file at '{}'".format(flags['resume_from_file']))
            optimizer_state_dict = model.load_model(flags['resume_from_file'])
            # optimizer_gen.load_state_dict(optimizer_state_dict)
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags['resume_from_file'], start_iteration))
            # for i in range(start_iteration):
                # optimizer_gen.step()
                # scheduler_gen.step()

        if num_gpus > 1:
            from torch.nn.parallel import DataParallel
            dp_model = DataParallel(model, device_ids = [i for i in range(num_gpus)])
            parallel = True
        else:
            parallel = False

        start_iteration = 1
        best_iteration = 1
        best_accuracy = 0
        best_exact_match = 0
        best_loss = float('inf')
        epoch_idx = 0

        logger.info("Training starts..")
        training_iteration = start_iteration
        while training_iteration < max_training_iterations:
            epoch_loss = 0
            epoch_gan_loss = 0
            epoch_acc = 0
            batch_num = 0
            epoch_idx += 1

            for sample_dict in train_loader:
                batch_num += 1
                input_batch = sample_dict['input_batch']
                token_type_ids = sample_dict['token_type_batch']
                input_lengths = sample_dict['input_length']
                derivation_representation_batch = sample_dict['derivation_representation_batch']
                situation_batch = sample_dict['situation_batch']
                situation_representation_batch = sample_dict['situation_representation_batch']
                labels = sample_dict['labels']
                choice_len_batch = sample_dict['choice_len_batch']
                if loss_type == 'reg':
                    choice_len_batch = choice_len_batch.float()


                is_best = False
                if parallel:
                    dp_model.module.train()
                else:
                    model.train()
                length_predictor.train()


                # Forward pass.
                if training_type == 'mc_only':                    
                    # logits.retain_grad()
                    if parallel:
                        logits, choice_hidden, loss = dp_model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, labels = labels)
                        gen_loss = torch.mean(loss)
                    else:
                        logits, choice_hidden, loss = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, labels = labels)
                        gen_loss = loss
                    # gen_loss = model.get_loss(logits, labels)

                    optimizer_gen.zero_grad()
                    gen_loss.backward()

                    # gen_loss.backward(retain_graph = True)

                    optimizer_gen.step()
                    scheduler_gen.step()

                    epoch_loss += gen_loss.item()

                    # choice_hidden = choice_hidden.detach()
                    # length_pred = length_predictor(choice_hidden)
                    # disc_loss = length_predictor.get_loss(length_pred, choice_len_batch)

                    # optimizer_disc.zero_grad()
                    # disc_loss.backward()
                    # optimizer_disc.step()
                    # scheduler_disc.step()
                elif training_type == 'gan':
                    if parallel:
                        logits, choice_hidden, loss = dp_model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, labels = labels)        
                        mc_loss = torch.mean(loss)            
                    else:
                        logits, choice_hidden, loss = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, labels = labels)
                        mc_loss = loss
                    length_pred = length_predictor(choice_hidden)
                    if loss_type == 'reg':
                        disc_loss = length_predictor.get_loss(length_pred, choice_len_batch)
                    elif loss_type == 'cls':
                        batch_size, choice_num, class_num = length_pred.size()
                        length_pred = length_pred.view(-1, class_num)
                        if maximize_tgt == 'entropy':
                            disc_loss = entropy(length_pred, )
                        elif maximize_tgt == 'disc_loss':
                            disc_loss = length_predictor.get_loss(length_pred, choice_len_batch)

                    gen_loss = mc_loss - adv_beta * disc_loss
                    epoch_loss += mc_loss.item()
                    epoch_gan_loss += disc_loss.item()

                    optimizer_gen.zero_grad()
                    gen_loss.backward()
                    optimizer_gen.step()
                    scheduler_gen.step()

                    choice_hidden = choice_hidden.detach()
                    length_pred = length_predictor(choice_hidden)
                    disc_loss = length_predictor.get_loss(length_pred, choice_len_batch)

                    optimizer_disc.zero_grad()
                    disc_loss.backward()
                    optimizer_disc.step()
                    scheduler_disc.step()

                elif training_type == 'adv_only':
                    if parallel:
                        logits, choice_hidden = dp_model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )
                    else:
                        logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )
                    choice_hidden = choice_hidden.detach()
                    length_pred = length_predictor(choice_hidden)
                    disc_loss = length_predictor.get_loss(length_pred, choice_len_batch)

                    optimizer_disc.zero_grad()
                    disc_loss.backward()
                    optimizer_disc.step()
                    scheduler_disc.step()
                else:
                    raise NotImplementedError


                loss, acc, recall, precision, F1 = model.get_metrics(logits, labels)
                # epoch_acc += acc

                # choice_list = sample_dict['choice_list'][0]
                # Print current metrics.
                if training_iteration % print_every == 0:
                    learning_rate = scheduler_gen.get_lr()[0]
                    disc_lr = scheduler_disc.get_lr()[0]
                    if training_type == 'gan':
                        if loss_type == 'reg':
                            l1_distance = length_predictor.get_metrics(length_pred, choice_len_batch)
                            logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f, l1 distance %.3f, learning_rate %.5f"% (training_iteration, loss, acc, l1_distance,
                                                                            learning_rate, ))
                        elif loss_type == 'cls':
                            length_cls_acc = length_predictor.get_metrics(length_pred, choice_len_batch)
                            logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f, length acc %.3f, learning_rate %.5f"% (training_iteration, loss, acc, length_cls_acc,
                                                                            learning_rate, ))
                    elif training_type == 'mc_only':
                        logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f,  learning_rate %.5f"% (training_iteration, loss, acc,
                                                                        learning_rate, ))
                    elif training_type == 'adv_only':
                        if loss_type == 'reg':
                            l1_distance = length_predictor.get_metrics(length_pred, choice_len_batch)
                            logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f, l1 distance %.3f, learning_rate %.5f"% (training_iteration, loss, acc, l1_distance,
                                                                            disc_lr, ))
                        elif loss_type == 'cls':
                            length_cls_acc = length_predictor.get_metrics(length_pred, choice_len_batch)
                            logger.info("Iteration %08d, loss %8.4f, accuracy %5.3f, length acc %.3f, learning_rate %.5f"% (training_iteration, loss, acc, length_cls_acc,
                                                                            disc_lr, ))



                # Evaluate on test set.
                if training_iteration % evaluate_every == 0:
                    with torch.no_grad():
                        if parallel:
                            dp_model.module.eval()
                        else:
                            model.eval()
                        length_predictor.eval()
                        logger.info("Evaluating on training set..")
                        
                        corr_total = 0
                        total = 0
                        all_pred = []
                        all_orig = []
                        total_l1 = 0
                        batch_num = 0
                        for sample_dict in eval_train_loader:
                            input_batch = sample_dict['input_batch']
                            token_type_ids = sample_dict['token_type_batch']
                            input_lengths = sample_dict['input_length']
                            derivation_representation_batch = sample_dict['derivation_representation_batch']
                            situation_batch = sample_dict['situation_batch']
                            situation_representation_batch = sample_dict['situation_representation_batch']
                            labels = sample_dict['labels']
                            choice_len_batch = sample_dict['choice_len_batch']
                            if parallel:
                                logits, choice_hidden = dp_model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )                            
                            else:
                                logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )
                            if training_type == 'gan' or training_type == 'adv_only':                 
                                length_pred = length_predictor(choice_hidden)
                                l1_distance = length_predictor.get_metrics(length_pred, choice_len_batch)
                                total_l1 += l1_distance

                            pred_labels = torch.argmax(logits, dim = 1)
                            corr = (pred_labels == labels).sum().item()
                            corr_total += corr
                            total += input_batch.size(0)
                            all_pred.append(pred_labels)
                            all_orig.append(labels)

                            batch_num += 1

                        avg_l1 = total_l1 / batch_num

                        all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
                        all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
                        accuracy = corr_total / total
                        
                        if training_type == 'gan' or training_type == 'adv_only':
                            if loss_type == 'reg':
                                logger.info("  Training Evaluation Accuracy: %5.3f,  L1: %.3f" % (accuracy, avg_l1))
                            elif loss_type == 'cls':
                                logger.info("  Training Evaluation Accuracy: %5.3f, length acc: %.3f" % (accuracy, avg_l1))
                        else:
                            logger.info("  Training Evaluation Accuracy: %5.3f" % (accuracy))

                    with torch.no_grad():
                        if parallel:
                            dp_model.module.eval()
                        else:
                            model.eval()
                        length_predictor.eval()
                        logger.info("Evaluating on validation set..")
                        
                        corr_total = 0
                        total = 0
                        all_pred = []
                        all_orig = []
                        total_l1 = 0
                        batch_num = 0
                        for sample_dict in valid_loader:
                            input_batch = sample_dict['input_batch']
                            token_type_ids = sample_dict['token_type_batch']
                            input_lengths = sample_dict['input_length']
                            derivation_representation_batch = sample_dict['derivation_representation_batch']
                            situation_batch = sample_dict['situation_batch']
                            situation_representation_batch = sample_dict['situation_representation_batch']
                            labels = sample_dict['labels']
                            choice_len_batch = sample_dict['choice_len_batch']
                            if parallel:
                                logits, choice_hidden = dp_model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )                            
                            else:
                                logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )
                            if training_type == 'gan' or training_type == 'adv_only':                 
                                length_pred = length_predictor(choice_hidden)
                                l1_distance = length_predictor.get_metrics(length_pred, choice_len_batch)
                                total_l1 += l1_distance

                            pred_labels = torch.argmax(logits, dim = 1)
                            corr = (pred_labels == labels).sum().item()
                            corr_total += corr
                            total += input_batch.size(0)
                            all_pred.append(pred_labels)
                            all_orig.append(labels)

                            batch_num += 1

                        avg_l1 = total_l1 / batch_num

                        all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
                        all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
                        accuracy = corr_total / total
                        
                        if training_type == 'gan' or training_type == 'adv_only':
                            if loss_type == 'reg':
                                logger.info("  Evaluation Accuracy: %5.3f, L1: %.3f" % (accuracy, avg_l1))
                            elif loss_type == 'cls':
                                logger.info("  Evaluation Accuracy: %5.3f, length acc: %.3f" % (accuracy, avg_l1))
                        else:
                            logger.info("  Evaluation Accuracy: %5.3f" % (accuracy))


                        if accuracy > best_accuracy:
                            is_best = True
                            best_accuracy = accuracy
                            best_recall = recall
                            best_precision = precision
                            best_F1 = F1
                            if parallel:
                                dp_model.module.update_state(acc=best_accuracy, F1 = best_F1, recall = best_recall, precision = best_precision, is_best=is_best)
                            else:
                                model.update_state(acc=best_accuracy, F1 = best_F1, recall = best_recall, precision = best_precision, is_best=is_best)
                        file_name = "checkpoint_iter{}.pth.tar".format(str(training_iteration))
                        if is_best:
                            if parallel:
                                dp_model.module.save_checkpoint(file_name=file_name, is_best=is_best,
                                                    optimizer_state_dict=optimizer_gen.state_dict())
                            else:
                                model.save_checkpoint(file_name=file_name, is_best=is_best,
                                                    optimizer_state_dict=optimizer_gen.state_dict())
                            if training_type == 'gan' or training_type == 'adv_only':
                                predictor_save_dir = os.path.join(flags['output_directory'], "length_predictor.pth")
                                predictor_state = {'state_dict': length_predictor.state_dict(), 'optimizer_state_dict': optimizer_disc.state_dict()}
                                torch.save(predictor_state, predictor_save_dir)

                training_iteration += 1
                if training_iteration > max_training_iterations:
                    break
        
            # epoch_loss /= batch_num
            # epoch_gan_loss /= batch_num
            # epoch_acc /= batch_num
            # logger.info("epoch {}: epoch loss: {:.3f},  epoch acc: {:.3f}, epoch gan loss: {:.3f}".format(epoch_idx, epoch_loss, epoch_acc, epoch_gan_loss))

        logger.info("Finished training.")




    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))
            test_dataset = MCGScanDataset(data_path, flags["data_directory"], split = split, input_vocabulary_file=flags["input_vocab_path"],
                                            target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False, k=flags["k"], 
                                            length_control = flags['length_control'], 
                                            contrast_size= flags['contrast_size'], 
                                            contrast_from_batch_size = flags['contrast_from_batch_size'],
                                            max_examples = flags['max_testing_examples'],
                                            less_length_label = False,
                                            max_len = 20,
                                            min_len = 0,
                                            )
            test_loader = DataLoader(test_dataset, batch_size = flags['test_batch_size'], shuffle = False, collate_fn = test_dataset.collate_fn,)


            logger.info("Done Loading {} dataset split.".format(split))
            logger.info("  Loaded {} examples.".format(test_dataset.num_examples))
            logger.info("  Input vocabulary size: {}".format(test_dataset.input_vocabulary_size))
            logger.info("  Most common input words: {}".format(test_dataset.input_vocabulary.most_common(5)))
            logger.info("  Output vocabulary size: {}".format(test_dataset.target_vocabulary_size))
            logger.info("  Most common target words: {}".format(test_dataset.target_vocabulary.most_common(5)))

            model_type = flags['model_type']

            if model_type == 'mc':
                model = DiscriminatorMCAdv(input_vocabulary_size=test_dataset.input_vocabulary_size,
                                            target_vocabulary_size=test_dataset.target_vocabulary_size,
                                            num_cnn_channels=test_dataset.image_channels,
                                            input_padding_idx=test_dataset.input_vocabulary.pad_idx,
                                            target_pad_idx=test_dataset.target_vocabulary.pad_idx,
                                            target_eos_idx=test_dataset.target_vocabulary.eos_idx,
                                            **flags)
            elif model_type == 'mc_v2':
                model = DiscriminatorMC_V2(input_vocabulary_size=test_dataset.input_vocabulary_size,
                                            target_vocabulary_size=test_dataset.target_vocabulary_size,
                                            num_cnn_channels=test_dataset.image_channels,
                                            input_padding_idx=test_dataset.input_vocabulary.pad_idx,
                                            target_pad_idx=test_dataset.target_vocabulary.pad_idx,
                                            target_eos_idx=test_dataset.target_vocabulary.eos_idx,
                                            **flags)            
            elif model_type == 'mc_v3':
                model = DiscriminatorMC_V3(input_vocabulary_size=test_dataset.input_vocabulary_size,
                                            target_vocabulary_size=test_dataset.target_vocabulary_size,
                                            num_cnn_channels=test_dataset.image_channels,
                                            input_padding_idx=test_dataset.input_vocabulary.pad_idx,
                                            target_pad_idx=test_dataset.target_vocabulary.pad_idx,
                                            target_eos_idx=test_dataset.target_vocabulary.eos_idx,
                                        **flags)  

            model = model.cuda() if use_cuda else model

            # Load model and vocabularies if resuming.
            assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
            logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
            model.load_model(flags["resume_from_file"])
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))

            with torch.no_grad():
                model.eval()
                logger.info("Evaluating...")
                
                corr_total = 0
                total = 0
                all_pred = []
                all_orig = []
                batch_num = 0
                # while True:
                #     for sample_dict in test_loader:
                #         input_batch = sample_dict['input_batch']
                #         token_type_ids = sample_dict['token_type_batch']
                #         input_lengths = sample_dict['input_length']
                #         derivation_representation_batch = sample_dict['derivation_representation_batch']
                #         situation_batch = sample_dict['situation_batch']
                #         situation_representation_batch = sample_dict['situation_representation_batch']
                #         labels = sample_dict['labels']
                #         choice_len_batch = sample_dict['choice_len_batch']
                #         logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )

                #         pred_labels = torch.argmax(logits, dim = 1)
                #         corr = (pred_labels == labels).sum().item()
                #         corr_total += corr
                #         total += input_batch.size(0)
                #         all_pred.append(pred_labels)
                #         all_orig.append(labels)

                #         batch_num += 1
                        
                #         pred_labels = torch.argmax(logits, dim = -1)
                #         logits = logits.view(-1)
                #         pred_labels = pred_labels.view(-1)
                #         target = labels.view(-1)
                #         print("logits: ", logits)
                #         # print("grad:   ", logits.grad)
                #         print("pred:   ", pred_labels, )
                #         print("target: ", target)
                #         print("target logits: ", logits[target.item()])
                #         # print("loss:   ", gen_loss.item())
                #         print(input_batch[0][target.item()].cpu())
                #         print(input_batch[0][pred_labels.item()].cpu())
                #         pause = input("???")


                for sample_dict in test_loader:
                    input_batch = sample_dict['input_batch']
                    token_type_ids = sample_dict['token_type_batch']
                    input_lengths = sample_dict['input_length']
                    derivation_representation_batch = sample_dict['derivation_representation_batch']
                    situation_batch = sample_dict['situation_batch']
                    situation_representation_batch = sample_dict['situation_representation_batch']
                    labels = sample_dict['labels']
                    choice_len_batch = sample_dict['choice_len_batch']
                    logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, situations_input = situation_batch, )

                    pred_labels = torch.argmax(logits, dim = 1)
                    corr = (pred_labels == labels).sum().item()
                    corr_total += corr
                    total += input_batch.size(0)
                    all_pred.append(pred_labels)
                    all_orig.append(labels)


                all_pred = torch.cat(all_pred, dim = 0).detach().cpu().numpy()
                all_orig = torch.cat(all_orig, dim = 0).detach().cpu().numpy()
                accuracy = corr_total / total
                
                logger.info("  Evaluation Accuracy: %5.3f" % (accuracy))
    
            
            # output_file = predict_and_save_disc(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
            
            # logger.info("Saved predictions to {}".format(output_file))
    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
