import argparse
import logging
import os
import pickle
import torch

from seq2seq.gSCAN_dataset import GroundedScanDataset, GroundedScanDatasetContrast, GroundedScanDatasetForTest
from seq2seq.helpers import sequence_accuracy, cal_recall, stat_prob_list, strip_pad
from seq2seq.model import Model, Model_CPGDecoder, ModelTransformer
from seq2seq.my_model import Discriminator, DiscriminatorMultipleChoice
from seq2seq.train import train
from seq2seq.predict import (predict_and_save, predict_and_save_beamsearch, predict_and_save_noeos, predict_and_save_transformer, 
                                predict_and_save_noeos_transformer, predict_and_save_lenreg, predict_and_save_eos_distribution, predict_beamsearch, predict_beamsearch_disc, predict_beamsearch_large, predict_beamsearch_stat, predict_disc, predict_sampling)
import torch.nn.functional as F

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
parser.add_argument("--weight_target_loss", type=float, default=0.3, help="Only used if --auxiliary_task set.")

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
parser.add_argument("--auxiliary_task", dest="auxiliary_task", default=False, action="store_true",
                    help="If set to true, the model predicts the target location from the joint attention over the "
                         "input instruction and world state.")
parser.add_argument("--no_auxiliary_task", dest="auxiliary_task", default=True, action="store_false")

# Command Encoder arguments
parser.add_argument("--embedding_dimension", type=int, default=25)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--encoder_hidden_size", type=int, default=100)
parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")
parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

# Decoder arguments
parser.add_argument("--num_decoder_layers", type=int, default=1)
parser.add_argument("--attention_type", type=str, default='bahdanau', choices=['bahdanau', 'luong'],
                    help="Luong not properly implemented.")
parser.add_argument("--decoder_dropout_p", type=float, default=0.3, help="Dropout on decoder embedding and LSTM.")
parser.add_argument("--decoder_hidden_size", type=int, default=100)
parser.add_argument("--conditional_attention", dest="conditional_attention", default=True, action="store_true",
                    help="If set to true joint attention over the world state conditioned on the input instruction is"
                         " used.")
parser.add_argument("--no_conditional_attention", dest="conditional_attention", default=False, action="store_false")

# Other arguments
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_type", type=str, default='mc', choices=['cls', 'mc'],)
parser.add_argument("--inf_type", type=str, default='orig', choices=['orig', 'no_eos', 'len_reg','eos_distribution', 'beam_search'],)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--aug_prob", type=float, default=0.7)
parser.add_argument("--white_portion", type=float, default=0.5)
parser.add_argument("--aug_strategy", type=str, default='rand', choices=['rand', 'fixed'],)
parser.add_argument("--insertion", type=str, default='mid', choices=['start', 'mid'],)
parser.add_argument("--max_white_num", type=int, default=5)
parser.add_argument("--num_transformer_layers", type=int, default=6)
parser.add_argument('--warmup', type=float, default=20000)
parser.add_argument("--warmup_lr", type=float, default=1e-6)
parser.add_argument("--recall_only", action = 'store_true')
parser.add_argument("--stat", action = 'store_true')


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

    if flags["test_batch_size"] > 1:
        raise NotImplementedError("Test batch size larger than 1 not implemented.")

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        raise NotImplementedError()
    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))
            max_len = 48
            min_len = 36
            beam_size = 8
            test_set = GroundedScanDatasetForTest(data_path, flags["data_directory"], split=split,
                                           input_vocabulary_file=flags["input_vocab_path"],
                                           target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                           k=flags["k"],
                                           max_len = max_len,
                                           min_len = min_len,
                                           )
            # test_set = GroundedScanDataset(data_path, flags["data_directory"], split='dev',
            #                                input_vocabulary_file=flags["input_vocab_path"],
            #                                target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
            #                                k=flags["k"],
            #                                )

            # test_set.read_dataset(max_examples=None,
            # print(flags["simple_situation_representation"])
            # pause = input("???")
            test_set.read_dataset(max_examples = flags['max_testing_examples'],
                                  simple_situation_representation=flags["simple_situation_representation"])
            logger.info("Done Loading {} dataset split.".format(flags["split"]))
            logger.info("  Loaded {} examples.".format(test_set.num_examples))
            logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
            logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
            logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
            logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

            # print(test_set.image_channels)
            # pause = input("???")

            if flags['model_type'] == 'cpg':
                model = Model_CPGDecoder(input_vocabulary_size=test_set.input_vocabulary_size,
                                        target_vocabulary_size=test_set.target_vocabulary_size,
                                        num_cnn_channels=test_set.image_channels,
                                        input_padding_idx=test_set.input_vocabulary.pad_idx,
                                        target_pad_idx=test_set.target_vocabulary.pad_idx,
                                        target_eos_idx=test_set.target_vocabulary.eos_idx,
                                        **flags)
            elif flags['model_type'] == 'transformer':
                model = ModelTransformer(input_vocabulary_size=test_set.input_vocabulary_size,
                                        target_vocabulary_size=test_set.target_vocabulary_size,
                                        num_cnn_channels=test_set.image_channels,
                                        input_padding_idx=test_set.input_vocabulary.pad_idx,
                                        target_pad_idx=test_set.target_vocabulary.pad_idx,
                                        target_eos_idx=test_set.target_vocabulary.eos_idx,
                                        **flags)
            else:
                model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
                            target_vocabulary_size=test_set.target_vocabulary_size,
                            num_cnn_channels=test_set.image_channels,
                            input_padding_idx=test_set.input_vocabulary.pad_idx,
                            target_pad_idx=test_set.target_vocabulary.pad_idx,
                            target_eos_idx=test_set.target_vocabulary.eos_idx,
                            **flags)

            disc_test_set = GroundedScanDatasetContrast(data_path, "data/target_length_split/", split=split,
                                           input_vocabulary_file="training_input_vocab_white.txt",
                                           target_vocabulary_file="training_target_vocab_white.txt", generate_vocabulary=False,
                                           k=flags["k"], 
                                           aug_prob = flags['aug_prob'], white_portion = flags['white_portion'],
                                           max_white_num = flags['max_white_num'], insertion = flags['insertion'], aug_strategy = flags['aug_strategy'],
                                           )

            flags['encoder_hidden_size'] = 128
            if flags['model_type'] == 'cls':
                disc_model = Discriminator(input_vocabulary_size=disc_test_set.input_vocabulary_size,
                            target_vocabulary_size=disc_test_set.target_vocabulary_size,
                            num_cnn_channels=test_set.image_channels,
                            input_padding_idx=disc_test_set.input_vocabulary.pad_idx,
                            target_pad_idx=disc_test_set.target_vocabulary.pad_idx,
                            target_eos_idx=disc_test_set.target_vocabulary.eos_idx,
                            **flags)
            elif flags['model_type'] == 'mc':
                disc_model = DiscriminatorMultipleChoice(input_vocabulary_size=disc_test_set.input_vocabulary_size,
                            target_vocabulary_size=disc_test_set.target_vocabulary_size,
                            num_cnn_channels=test_set.image_channels,
                            input_padding_idx=disc_test_set.input_vocabulary.pad_idx,
                            target_pad_idx=disc_test_set.target_vocabulary.pad_idx,
                            target_eos_idx=disc_test_set.target_vocabulary.eos_idx,
                            **flags)                


            model = model.cuda() if use_cuda else model
            disc_model = disc_model.cuda()
            device = torch.device("cuda")

            # model = model.cuda() if use_cuda else model
            # disc_model = disc_model.cuda()

            # device = torch.device("cpu")

            # Load model and vocabularies if resuming.
            assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
            logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
            model.load_model(flags["resume_from_file"])

            if flags['model_type'] == 'cls':
                disc_model.load_model("checkpoints/disc_layer6/model_best.pth.tar")
            elif flags['model_type'] == 'mc':
                disc_model.load_model("checkpoints/disc_mc_20_20_layer6/model_best.pth.tar")
                # disc_model.load_model("checkpoints/disc_mc_20_20_continue/model_best.pth.tar")
                # disc_model.load_model("checkpoints/disc_mc_20_20/model_best.pth.tar")
                # disc_model.load_model("checkpoints/disc_mc_2_2/model_best.pth.tar")

            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
            # output_file_name = "_".join([split, flags["output_file_name"]])
            # output_file_path = os.path.join(flags["output_directory"], output_file_name)
            # logger.info("Saved predictions to {}".format(output_file))

            test_batch_size = 30
            data_iterator = test_set.get_data_iterator(batch_size=test_batch_size)
            # flags.pop('max_examples_to_evaluate')
            # for result in predict_sampling(data_iterator = data_iterator, model = model, max_examples_to_evaluate = 10, 
            #                                 max_decoding_steps=flags['max_decoding_steps'],
            #                                 pad_idx=test_set.target_vocabulary.pad_idx, sos_idx=test_set.target_vocabulary.sos_idx,
            #                                 eos_idx=test_set.target_vocabulary.eos_idx):
            
            correct = 0
            total = 0
            res_dict = {}
            recall = 0
            recall_dict = {}
            count = 0
            times = 5
            
            all_prob_before, all_prob_after = [], []
            for data_batch in tqdm(data_iterator):
                count += 1
                if count % 10 == 0:
                    print(count * test_batch_size)
                # result_list = predict_beamsearch_large(data_batch = data_batch, model = model,  
                #                                     max_decoding_steps=flags['max_decoding_steps'],
                #                                     pad_idx=test_set.target_vocabulary.pad_idx, sos_idx=test_set.target_vocabulary.sos_idx,
                #                                     eos_idx=test_set.target_vocabulary.eos_idx,
                #                                     beam_size = 5,
                #                                     max_length = 25,
                #                                     times = times,
                #                                     )
                # with open("./saved_res/greedy_decode_res_times%d_iter%d"%(times, count), 'wb') as f:
                #     pickle.dump(result_list, f)
                result_list = predict_beamsearch_disc(data_batch = data_batch, model = model,  
                                                    max_decoding_steps=flags['max_decoding_steps'],
                                                    pad_idx=test_set.target_vocabulary.pad_idx, sos_idx=test_set.target_vocabulary.sos_idx,
                                                    eos_idx=test_set.target_vocabulary.eos_idx,
                                                    beam_size = beam_size,
                                                    max_length = max_len,
                                                    min_length = min_len,
                                                    )
                
                if flags['stat']:
                    prob_before, prob_after = predict_beamsearch_stat(data_batch = data_batch, model = model,  
                                                                        max_decoding_steps=flags['max_decoding_steps'],
                                                                        pad_idx=test_set.target_vocabulary.pad_idx, sos_idx=test_set.target_vocabulary.sos_idx,
                                                                        eos_idx=test_set.target_vocabulary.eos_idx,
                                                                        beam_size = 5,
                                                                        max_length = max_len,
                                                                        )
                    
                    all_prob_before += prob_before
                    all_prob_after += prob_after
                    result_list = None
                    continue
                # result_list = predict_beamsearch(data_batch = data_batch, model = model,  
                #                                     max_decoding_steps=flags['max_decoding_steps'],
                #                                     pad_idx=test_set.target_vocabulary.pad_idx, sos_idx=test_set.target_vocabulary.sos_idx,
                #                                     eos_idx=test_set.target_vocabulary.eos_idx,
                #                                     beam_size = 5,
                #                                     max_length = 15,
                #                                     )
                for result in result_list:
                    input_sequence, derivation_spec, situation_spec, output_sequence, target_sequence, situation = result
                    num_sampling = len(output_sequence)
                    input_sequence = strip_pad(input_sequence[1:-1], pad = test_set.input_vocabulary.pad_idx, eos = test_set.input_vocabulary.eos_idx)
                    target_sequence = strip_pad(target_sequence[1:-1], pad = test_set.target_vocabulary.pad_idx, eos = test_set.target_vocabulary.eos_idx)
                    input_x = test_set.array_to_sentence(input_sequence.tolist(), vocabulary = 'input')
                    target_y = test_set.array_to_sentence(target_sequence.tolist(), vocabulary = 'target')
                    all_pred = [test_set.array_to_sentence(x, vocabulary = 'target') for x in output_sequence]
                    target_len = len(target_y)
                    recall_flag = cal_recall(target_y, all_pred)
                    recall += recall_flag
                    # print(target_y)
                    # print(all_pred)
                    # print(recall_flag)
                    # pause = input("???")
                    total += 1
                    if target_len not in recall_dict:
                        recall_dict[target_len] = [0,1]
                        recall_dict[target_len][0] += recall_flag
                    else:
                        recall_dict[target_len][0] += recall_flag
                        recall_dict[target_len][1] += 1
                    if flags['recall_only']:
                        continue




                    input_array = disc_test_set.sentence_to_array(input_x, vocabulary = 'input')
                    input_array = [torch.tensor(input_array, dtype = torch.long, device = device).unsqueeze(0) for _ in range(num_sampling)]
                    target_array = [torch.tensor(disc_test_set.sentence_to_array(x, vocabulary = 'target'), dtype = torch.long, device = device).unsqueeze(0) for x in all_pred]
                    
                    max_target_len = max([len(x) for x in all_pred])
                    input_len = len(input_x)
                    max_length = input_len + max_target_len + 3

                    situation_array = [situation.detach().clone() for _ in range(num_sampling)]
                    
                    input_batch = []
                    token_type_batch = []

                    for idx in range(num_sampling):
                        input_x = torch.cat([input_array[idx],target_array[idx][:, 1:]], dim = 1)
                        sen1_len = input_array[idx].size(1)
                        sen2_len = target_array[idx].size(1) - 1
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
                        input_batch.append(padded_input)                 
                        token_type_batch.append(token_type_tensor)
                    input_batch = torch.cat(input_batch, dim = 0)
                    token_type_batch = torch.cat(token_type_batch, dim = 0)

                    if flags['model_type'] == 'cls':
                        situation_batch = torch.stack(situation_array, dim = 0)  ## TODO: check the shape
                    else:
                        situation_batch = situation.unsqueeze(0)
                        input_batch = input_batch.unsqueeze(0)
                        token_type_batch == token_type_batch.unsqueeze(0)


                    logits = disc_model(input_batch, token_type_batch, None, situation_batch)
                    
                    if flags['model_type'] == 'cls':
                
                        logits = F.softmax(logits, dim = 1)
                        corr_logits = logits[:,1]
                        # top_10 = torch.topk(corr_logits,k = 5).indices
                        top_indices = torch.argsort(-corr_logits)
                        # print("top10 score: ", corr_logits[top_indices])
                        for top_idx in top_indices:
                            curr_output = all_pred[top_idx]
                            # print("prediction probability: ", corr_logits[top_idx])
                            # print("output: ", curr_output, " (%d)"%(len(curr_output)))
                            # print("ground truth: ", target_y)
                            acc = sequence_accuracy(curr_output, target_y)
                            # print("acc: ", acc)
                        top1_idx = top_indices[0]
                        curr_output = all_pred[top1_idx]
                        acc = sequence_accuracy(curr_output, target_y)

                    elif flags['model_type'] == 'mc':
                        top1_idx = torch.argmax(logits, dim = 1).view(-1)
                        curr_output = all_pred[top1_idx]
                        acc = sequence_accuracy(curr_output, target_y)


                    if target_len not in res_dict:
                        res_dict[target_len] = [0,0]

                    res_dict[target_len][1] += 1
                    if acc == 100:  
                        correct += 1
                        res_dict[target_len][0] += 1
                        # print("!!")
                    # for idx in range(len(all_pred)):
                    #     curr_output = all_pred[idx]
                    #     print("prediction probability: ", corr_logits[idx])
                    #     acc = sequence_accuracy(curr_output, target_y)
                    #     print("acc: ", acc, " prob: ", corr_logits[idx])

            if flags['stat']:
                before_stat = stat_prob_list(all_prob_before)
                after_stat = stat_prob_list(all_prob_after)
                print(before_stat)
                print(after_stat)
                with open("saved_results/probability_dist_%d-%d.pkl"%(min_len, max_len), 'wb') as f:
                    pickle.dump((before_stat, after_stat), f)
                return

            print("recall: ", recall)
            print("total: ", total)
            for k,v in recall_dict.items():
                print(k,v)
                with open("saved_results/recall_beam%d_%d-%d"%(beam_size, min_len, max_len), 'wb') as f:
                    pickle.dump(recall_dict, f)

            if not flags['recall_only']:
                print("correct: ", correct)
                print("total: ", total)
                for k,v in res_dict.items():
                    print(k, v)

    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
