import logging
import torch
import os
from torch.optim.lr_scheduler import LambdaLR
from transformers import WarmUp

from seq2seq.my_model import DiscriminatorMC_V2, DiscriminatorMCAdv, DiscriminatorMultipleChoice, PredNet, PredNetTransformer, PredNetTransformerV2, Discriminator
from seq2seq.model import Model, Model_CPGDecoder, ModelTransformer
from seq2seq.my_model import ModelWithLengthBias
from seq2seq.tri_state_lr_scheduler import GradualWarmupScheduler

from seq2seq.gSCAN_dataset import (GroundedScanDataset, GroundedScanDatasetContrastV2, 
                                    GroundedScanDatasetContrastV4, GroundedScanDatasetPad, 
                                    GroundedScanDatasetContrast, MultipleChoiceGScan)
from seq2seq.helpers import log_parameters
from seq2seq.evaluate import (evaluate_length_pred, evaluate, evaluate_transformer, evaluate_withlen, 
                                evaluate_pad, evaluate_disc)

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, model_type = 'orig', 
          no_eos = False,
          **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k,
                                       no_eos = no_eos,
                                       )
    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = GroundedScanDataset(data_path, data_directory, split="dev",  # TODO: use dev set here
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0,
                                   no_eos = no_eos
                                   )
    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    print(training_set.input_vocabulary.pad_idx)
    print(training_set.target_vocabulary.pad_idx)

    if model_type == 'cpg':
        model = Model_CPGDecoder(input_vocabulary_size=training_set.input_vocabulary_size,
                                target_vocabulary_size=training_set.target_vocabulary_size,
                                num_cnn_channels=training_set.image_channels,
                                input_padding_idx=training_set.input_vocabulary.pad_idx,
                                target_pad_idx=training_set.target_vocabulary.pad_idx,
                                target_eos_idx=training_set.target_vocabulary.eos_idx,
                                **cfg)
    
    elif model_type == 'transformer':
        model = ModelTransformer(input_vocabulary_size=test_set.input_vocabulary_size,
                                target_vocabulary_size=test_set.target_vocabulary_size,
                                num_cnn_channels=test_set.image_channels,
                                input_padding_idx=test_set.input_vocabulary.pad_idx,
                                target_pad_idx=test_set.target_vocabulary.pad_idx,
                                target_eos_idx=test_set.target_vocabulary.eos_idx,
                                **cfg)
    else:
        model = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                    target_vocabulary_size=training_set.target_vocabulary_size,
                    num_cnn_channels=training_set.image_channels,
                    input_padding_idx=training_set.input_vocabulary.pad_idx,
                    target_pad_idx=training_set.target_vocabulary.pad_idx,
                    target_eos_idx=training_set.target_vocabulary.eos_idx,
                    **cfg)
    model = model.cuda() if use_cuda else model
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # Forward pass.
            target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            loss = model.get_loss(target_scores, target_batch)
            if auxiliary_task:
                target_loss = model.get_auxiliary_loss(target_position_scores, target_positions)
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model.get_metrics(target_scores, target_batch)
                if auxiliary_task:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")
                    if model_type == 'transformer':
                        accuracy, exact_match, target_accuracy = evaluate_transformer(
                            test_set.get_data_iterator(batch_size=1), model=model,
                            max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                            sos_idx=test_set.target_vocabulary.sos_idx,
                            eos_idx=test_set.target_vocabulary.eos_idx,
                            max_examples_to_evaluate=kwargs["max_testing_examples"])                        
                    else:
                        accuracy, exact_match, target_accuracy = evaluate(
                            test_set.get_data_iterator(batch_size=1), model=model,
                            max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                            sos_idx=test_set.target_vocabulary.sos_idx,
                            eos_idx=test_set.target_vocabulary.eos_idx,
                            max_examples_to_evaluate=kwargs["max_testing_examples"],
                            no_eos = no_eos)
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")


def pred_net_train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, model_type = 'orig',
          **kwargs):

    num_transformer_layers = kwargs.pop('num_transformer_layers')
    warmup = kwargs.pop('warmup')
    warmup_lr = kwargs.pop('warmup_lr')

    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k)
    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = GroundedScanDataset(data_path, data_directory, split="dev",  # TODO: use dev set here
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    print(training_set.input_vocabulary.pad_idx)
    print(training_set.target_vocabulary.pad_idx)

    if model_type == 'transformer':
        model = PredNetTransformer(input_vocabulary_size=test_set.input_vocabulary_size,
                                target_vocabulary_size=test_set.target_vocabulary_size,
                                num_cnn_channels=test_set.image_channels,
                                input_padding_idx=test_set.input_vocabulary.pad_idx,
                                target_pad_idx=test_set.target_vocabulary.pad_idx,
                                target_eos_idx=test_set.target_vocabulary.eos_idx,
                                **cfg)    

    elif model_type == 'transformer_v2':
        model = PredNetTransformerV2(input_vocabulary_size=test_set.input_vocabulary_size,
                                target_vocabulary_size=test_set.target_vocabulary_size,
                                num_cnn_channels=test_set.image_channels,
                                input_padding_idx=test_set.input_vocabulary.pad_idx,
                                target_pad_idx=test_set.target_vocabulary.pad_idx,
                                target_eos_idx=test_set.target_vocabulary.eos_idx,
                                **cfg)     
    else:
        model = PredNet(input_vocabulary_size=test_set.input_vocabulary_size,
                    target_vocabulary_size=test_set.target_vocabulary_size,
                    num_cnn_channels=test_set.image_channels,
                    input_padding_idx=test_set.input_vocabulary.pad_idx,
                    target_pad_idx=test_set.target_vocabulary.pad_idx,
                    target_eos_idx=test_set.target_vocabulary.eos_idx,
                    **cfg)
    model = model.cuda() if use_cuda else model
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    after_scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))
    
    scheduler = GradualWarmupScheduler(optimizer, 1.0, total_epoch = warmup, after_scheduler = after_scheduler)

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_l1 = 99999
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # print(target_lengths)
            # print(type(target_lengths))
            # pause = input("???")
            target_lengths = torch.from_numpy(target_lengths).to(input_batch).float()
            # Forward pass.
            pred_length = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            loss = model.get_loss(pred_length, target_lengths)

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                L1_distance = model.get_metrics(pred_length, target_lengths)
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, L1 distance %5.2f, lr %.5f" % (training_iteration, loss, L1_distance, learning_rate))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")

                    l2_distance, l1_distance, l2_list, l1_list, target_len_list = evaluate_length_pred(
                        test_set.get_data_iterator(batch_size=200), model=model,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"])
                    logger.info("  L2 distance: %5.2f L1 distance: %5.2f , learning_rate %.5f" % (l2_distance, l1_distance, learning_rate))
                    if l1_distance < best_l1:
                        is_best = True
                        best_l1 = l1_distance
                        model.update_state(l2 = l2_distance, l1=l1_distance, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")

def withlen_train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, model_type = 'orig', alpha = 0.01, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k)
    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = GroundedScanDataset(data_path, data_directory, split="dev",  # TODO: use dev set here
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    print(training_set.input_vocabulary.pad_idx)
    print(training_set.target_vocabulary.pad_idx)


    model = ModelWithLengthBias(input_vocabulary_size=training_set.input_vocabulary_size,
                            target_vocabulary_size=training_set.target_vocabulary_size,
                            num_cnn_channels=training_set.image_channels,
                            input_padding_idx=training_set.input_vocabulary.pad_idx,
                            target_pad_idx=training_set.target_vocabulary.pad_idx,
                            target_eos_idx=training_set.target_vocabulary.eos_idx,
                            alpha = alpha,
                            **cfg)
    model = model.cuda() if use_cuda else model
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # Forward pass.
            target_lengths = torch.from_numpy(target_lengths).to(input_batch).float()
            target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            loss = model.get_loss(target_scores, target_batch)
            if auxiliary_task:
                target_loss = model.get_auxiliary_loss(target_position_scores, target_positions)
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            alpha_value = model.attention_decoder.alpha.data.item()
            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model.get_metrics(target_scores, target_batch)
                if auxiliary_task:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " alpha %.4f" % (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, alpha_value))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")

                    accuracy, exact_match, target_accuracy = evaluate_withlen(
                        test_set.get_data_iterator(batch_size=1), model=model,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"])
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")


def pad_train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, model_type = 'orig', **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    aug_prob = kwargs.pop('aug_prob')
    white_portion = kwargs.pop('white_portion')
    max_white_num = kwargs.pop('max_white_num')
    insertion = kwargs.pop('insertion')
    aug_strategy = kwargs.pop("aug_strategy")
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDatasetPad(data_path, data_directory, split="train",
                                        input_vocabulary_file=input_vocab_path,
                                        target_vocabulary_file=target_vocab_path,
                                        generate_vocabulary=generate_vocabularies, k=k,
                                        aug_prob = aug_prob, white_portion = white_portion,
                                        max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                        )
    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))


    logger.info("Loading Dev. set...")
    test_set = GroundedScanDatasetPad(data_path, data_directory, split="dev",  # TODO: use dev set here
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0, aug_prob = -1,
                                   
                                   )
    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    print(training_set.input_vocabulary.pad_idx)
    print(training_set.target_vocabulary.pad_idx)


    model = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                target_vocabulary_size=training_set.target_vocabulary_size,
                num_cnn_channels=training_set.image_channels,
                input_padding_idx=training_set.input_vocabulary.pad_idx,
                target_pad_idx=training_set.target_vocabulary.pad_idx,
                target_eos_idx=training_set.target_vocabulary.eos_idx,
                **cfg)
    model = model.cuda() if use_cuda else model
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # Forward pass.
            target_scores, target_position_scores = model(commands_input=input_batch, commands_lengths=input_lengths,
                                                          situations_input=situation_batch, target_batch=target_batch,
                                                          target_lengths=target_lengths)
            loss = model.get_loss(target_scores, target_batch)
            if auxiliary_task:
                target_loss = model.get_auxiliary_loss(target_position_scores, target_positions)
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model.get_metrics(target_scores, target_batch)
                if auxiliary_task:
                    auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f"% (training_iteration, loss, accuracy, exact_match,
                                                                 learning_rate, ))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")

                    accuracy, exact_match, target_accuracy = evaluate_pad(
                        test_set.get_data_iterator(batch_size=1), model=model,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        white_idx = test_set.white_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"])
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f" % (accuracy, exact_match, target_accuracy))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = "checkpoint.pth.tar".format(str(training_iteration))
                    if is_best:
                        model.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")

def disc_train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
            target_vocab_path: str, training_batch_size: int, test_batch_size: int, cnn_kernel_size: int, cnn_dropout_p: float,
            cnn_hidden_num_channels: int, simple_situation_representation: bool,
            encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
            lr_decay_steps: int, resume_from_file: str, max_training_iterations: int, output_directory: str,
            print_every: int, evaluate_every: int, k: int, max_training_examples=None, seed=42, model_type = 'cls', **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    aug_prob = kwargs.pop('aug_prob')
    white_portion = kwargs.pop('white_portion')
    max_white_num = kwargs.pop('max_white_num')
    insertion = kwargs.pop('insertion')
    aug_strategy = kwargs.pop("aug_strategy")
    num_transformer_layers = kwargs.pop('num_transformer_layers')
    warmup = kwargs.pop('warmup')
    warmup_lr = kwargs.pop('warmup_lr')
    data_type = kwargs.pop('data_type')
    contrast_size = kwargs.pop("contrast_size")
    contrast_from_batch_size = kwargs.pop("contrast_from_batch_size")

    # multi_task = kwargs.pop("multi_task")


    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    if data_type == 'v1':
        training_set = GroundedScanDatasetContrast(data_path, data_directory, split="train",
                                                input_vocabulary_file=input_vocab_path,
                                                target_vocabulary_file=target_vocab_path,
                                                generate_vocabulary=generate_vocabularies, k=k,
                                                aug_prob = aug_prob, white_portion = white_portion,
                                                max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                                )
    elif data_type == 'v2':
        training_set = GroundedScanDatasetContrastV2(data_path, data_directory, split="train",
                                                input_vocabulary_file=input_vocab_path,
                                                target_vocabulary_file=target_vocab_path,
                                                generate_vocabulary=generate_vocabularies, k=k,
                                                aug_prob = aug_prob, white_portion = white_portion,
                                                max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                                )
    elif data_type == 'v4':
        training_set = GroundedScanDatasetContrastV4(data_path, data_directory, split="train",
                                                input_vocabulary_file=input_vocab_path,
                                                target_vocabulary_file=target_vocab_path,
                                                generate_vocabulary=generate_vocabularies, k=k,
                                                aug_prob = aug_prob, white_portion = white_portion,
                                                max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                                )
    elif data_type == 'mc':
        training_set = MultipleChoiceGScan(data_path, data_directory, split="train",
                                            input_vocabulary_file=input_vocab_path,
                                            target_vocabulary_file=target_vocab_path,
                                            generate_vocabulary=generate_vocabularies, k=k,
                                            aug_prob = aug_prob, white_portion = white_portion,
                                            max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                            contrast_size= contrast_size, contrast_from_batch_size = contrast_from_batch_size,
                                            )


    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))


    logger.info("Loading Dev. set...")
    if data_type == 'v1':
        test_set = GroundedScanDatasetContrast(data_path, data_directory, split="dev",  # TODO: use dev set here
                                    input_vocabulary_file=input_vocab_path,
                                    target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0, 
                                    aug_prob = aug_prob, white_portion = white_portion,
                                    max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                    )
    elif data_type == 'v2':
        test_set = GroundedScanDatasetContrastV2(data_path, data_directory, split="dev",  # TODO: use dev set here
                                    input_vocabulary_file=input_vocab_path,
                                    target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0, 
                                    aug_prob = aug_prob, white_portion = white_portion,
                                    max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                    )
    elif data_type == 'v4':
        test_set = GroundedScanDatasetContrastV4(data_path, data_directory, split="dev",  # TODO: use dev set here
                                    input_vocabulary_file=input_vocab_path,
                                    target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0, 
                                    aug_prob = aug_prob, white_portion = white_portion,
                                    max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                    )
    elif data_type == 'mc':
        test_set = MultipleChoiceGScan(data_path, data_directory, split="dev",  # TODO: use dev set here
                                    input_vocabulary_file=input_vocab_path,
                                    target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0, 
                                    aug_prob = aug_prob, white_portion = white_portion,
                                    max_white_num = max_white_num, insertion = insertion, aug_strategy = aug_strategy,
                                    contrast_size= contrast_size, contrast_from_batch_size = contrast_from_batch_size,
                                    )

    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")
    print(training_set.input_vocabulary.pad_idx)
    print(training_set.target_vocabulary.pad_idx)

    if model_type == 'cls':
        model = Discriminator(input_vocabulary_size=training_set.input_vocabulary_size,
                            target_vocabulary_size=training_set.target_vocabulary_size,
                            num_cnn_channels=training_set.image_channels,
                            input_padding_idx=training_set.input_vocabulary.pad_idx,
                            target_pad_idx=training_set.target_vocabulary.pad_idx,
                            target_eos_idx=training_set.target_vocabulary.eos_idx,
                            **cfg)
    elif model_type == 'mc':
        model = DiscriminatorMultipleChoice(input_vocabulary_size=training_set.input_vocabulary_size,
                                    target_vocabulary_size=training_set.target_vocabulary_size,
                                    num_cnn_channels=training_set.image_channels,
                                    input_padding_idx=training_set.input_vocabulary.pad_idx,
                                    target_pad_idx=training_set.target_vocabulary.pad_idx,
                                    target_eos_idx=training_set.target_vocabulary.eos_idx,
                                    **cfg)
    elif model_type == 'mc_v2':
        model = DiscriminatorMC_V2(input_vocabulary_size=training_set.input_vocabulary_size,
                                    target_vocabulary_size=training_set.target_vocabulary_size,
                                    num_cnn_channels=training_set.image_channels,
                                    input_padding_idx=training_set.input_vocabulary.pad_idx,
                                    target_pad_idx=training_set.target_vocabulary.pad_idx,
                                    target_eos_idx=training_set.target_vocabulary.eos_idx,
                                    **cfg)

    model = model.cuda() if use_cuda else model
    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))
    after_scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))
    
    scheduler = GradualWarmupScheduler(optimizer, 1.0, total_epoch = warmup, after_scheduler = after_scheduler)

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file:
        assert os.path.isfile(resume_from_file), "No checkpoint found at {}".format(resume_from_file)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file))
        optimizer_state_dict = model.load_model(resume_from_file)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file, start_iteration))
        for i in range(start_iteration):
            optimizer.step()
            scheduler.step()

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, token_type_ids, input_lengths, _, situation_batch, _, labels,
            agent_positions, target_positions) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model.train()

            # Forward pass.
            if model_type == 'mc_v2':
                logits, choice_hidden = model(commands_input=input_batch, token_type_ids = token_type_ids, commands_lengths=input_lengths,
                                situations_input=situation_batch, )                
            else:
                logits = model(commands_input=input_batch, token_type_ids = token_type_ids, commands_lengths=input_lengths,
                                situations_input=situation_batch, )
            loss = model.get_loss(logits, labels)
            target_loss = 0
            

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
                logger.info("Iteration %08d, loss %8.4f, accuracy %5.2f, learning_rate %.5f"% (training_iteration, loss, acc,
                                                                 learning_rate, ))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model.eval()
                    logger.info("Evaluating..")

                    if model_type == 'mc':
                        acc, recall, precision, F1 = evaluate_disc(
                            test_set.get_data_iterator(batch_size=test_batch_size), model=model,
                            max_examples_to_evaluate=kwargs["max_testing_examples"],
                            multi_class = True
                            )
                    
                    else:
                        acc, recall, precision, F1 = evaluate_disc(
                            test_set.get_data_iterator(batch_size=test_batch_size), model=model,
                            max_examples_to_evaluate=kwargs["max_testing_examples"])
                    logger.info("  Evaluation Accuracy: %5.2f, recall: %.3f, precision: %.3f, F1: %.3f" % (acc, recall, precision, F1))
                    if acc > best_accuracy:
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
