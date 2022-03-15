from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from typing import List
from typing import Dict
from typing import Tuple
import os
import shutil

from seq2seq.cnn_model import ConvolutionalNet
from seq2seq.cnn_model import DownSamplingConvolutionalNet
from seq2seq.seq2seq_model import EncoderRNN
from seq2seq.seq2seq_model import Attention
from seq2seq.my_seq2seq_model import LengthDecoder, AttentionDecoderWithLength

from allennlp.modules.transformer.bimodal_connection_layer import BiModalConnectionLayer
from allennlp.modules.util import replicate_layers

from transformers import ViTModel

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from sklearn.metrics import precision_score, recall_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


class PredNet(nn.Module):
    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, **kwargs):
        super(PredNet, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.visual_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=decoder_hidden_size,
                                          hidden_size=decoder_hidden_size)

        self.auxiliary_task = auxiliary_task
        if auxiliary_task:
            self.auxiliary_loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)
        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.enc_hidden_to_dec_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.textual_attention = Attention(key_size=encoder_hidden_size, query_size=decoder_hidden_size,
                                           hidden_size=decoder_hidden_size)

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]

        self.attention_decoder = LengthDecoder(hidden_size=decoder_hidden_size,
                                                dropout_probability=decoder_dropout_p,
                                                textual_attention=self.textual_attention,
                                                visual_attention=self.visual_attention,
                                                conditional_attention=conditional_attention,
                                                num_layers = num_decoder_layers)

        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.loss_criterion = nn.MSELoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            if reduction == 'mean':
                gap = torch.abs(pred_lengths - target_lengths).mean()
            else:
                gap = torch.abs(pred_lengths - target_lengths)
        return gap

    def get_loss(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        if reduction == 'mean':
            loss = self.loss_criterion(pred_lengths, target_lengths)
        else:
            loss = F.mse_loss(pred_lengths, target_lengths, reduction = 'none')
        return loss

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched = self.attention_decoder(input_tokens=target_batch,
                                                                        input_lengths=target_lengths,
                                                                        init_hidden=initial_hidden,
                                                                        encoded_commands=encoded_commands,
                                                                        commands_lengths=command_lengths,
                                                                        encoded_situations=encoded_situations)
        return decoder_output_batched

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                           situations_input=situations_input)
        decoder_output = self.decode_input_batched(
            target_batch=target_batch, target_lengths=target_lengths, initial_hidden=encoder_output["hidden_states"],
            encoded_commands=encoder_output["encoded_commands"]["encoder_outputs"], command_lengths=commands_lengths,
            encoded_situations=encoder_output["encoded_situations"])

        return decoder_output

    def update_state(self, is_best: bool, l2=None, l1=None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.l2 = l2
            self.l1 = l1
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class ModelWithLengthBias(nn.Module):
    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, alpha, **kwargs):
        super(ModelWithLengthBias, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        # Attention over the output features of the ConvolutionalNet.
        # Input: [bsz, 1, decoder_hidden_size], [bsz, image_width * image_width, cnn_hidden_num_channels * 3]
        # Output: [bsz, 1, decoder_hidden_size], [bsz, 1, image_width * image_width]
        self.visual_attention = Attention(key_size=cnn_hidden_num_channels * 3, query_size=decoder_hidden_size,
                                          hidden_size=decoder_hidden_size)

        self.auxiliary_task = auxiliary_task
        if auxiliary_task:
            self.auxiliary_loss_criterion = nn.NLLLoss()

        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.encoder = EncoderRNN(input_size=input_vocabulary_size,
                                  embedding_dim=embedding_dimension,
                                  rnn_input_size=embedding_dimension,
                                  hidden_size=encoder_hidden_size, num_layers=num_encoder_layers,
                                  dropout_probability=encoder_dropout_p, bidirectional=encoder_bidirectional,
                                  padding_idx=input_padding_idx)
        # Used to project the final encoder state to the decoder hidden state such that it can be initialized with it.
        self.enc_hidden_to_dec_hidden = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.textual_attention = Attention(key_size=encoder_hidden_size, query_size=decoder_hidden_size,
                                           hidden_size=decoder_hidden_size)

        # Input: [batch_size, max_target_length], initial hidden: ([batch_size, hidden_size], [batch_size, hidden_size])
        # Input for attention: [batch_size, max_input_length, hidden_size],
        #                      [batch_size, image_width * image_width, hidden_size]
        # Output: [max_target_length, batch_size, target_vocabulary_size]
        self.attention_type = attention_type
        self.attention_decoder = AttentionDecoderWithLength(hidden_size=decoder_hidden_size,
                                                                output_size=target_vocabulary_size,
                                                                num_layers=num_decoder_layers,
                                                                dropout_probability=decoder_dropout_p,
                                                                padding_idx=target_pad_idx,
                                                                textual_attention=self.textual_attention,
                                                                visual_attention=self.visual_attention,
                                                                conditional_attention=conditional_attention,
                                                                eos_idx = target_eos_idx, 
                                                                )

        self.penalty_alpha = alpha


        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.loss_criterion = nn.NLLLoss(ignore_index=target_pad_idx)
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    @staticmethod
    def remove_start_of_sequence(input_tensor: torch.Tensor) -> torch.Tensor:
        """Get rid of SOS-tokens in targets batch and append a padding token to each example in the batch."""
        batch_size, max_time = input_tensor.size()
        input_tensor = input_tensor[:, 1:]
        output_tensor = torch.cat([input_tensor, torch.zeros(batch_size, device=device, dtype=torch.long).unsqueeze(
            dim=1)], dim=1)
        return output_tensor

    def get_metrics(self, target_scores: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            targets = self.remove_start_of_sequence(targets)
            mask = (targets != self.target_pad_idx).long()
            total = mask.sum().data.item()
            predicted_targets = target_scores.max(dim=2)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long()
            match_targets = (equal_targets * mask)
            match_sum_per_example = match_targets.sum(dim=1)
            expected_sum_per_example = mask.sum(dim=1)
            batch_size = expected_sum_per_example.size(0)
            exact_match = 100. * (match_sum_per_example == expected_sum_per_example).sum().data.item() / batch_size
            match_targets_sum = match_targets.sum().data.item()
            accuracy = 100. * match_targets_sum / total
        return accuracy, exact_match

    @staticmethod
    def get_auxiliary_accuracy(target_scores: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            predicted_targets = target_scores.max(dim=1)[1]
            equal_targets = torch.eq(targets.data, predicted_targets.data).long().sum().data.item()
            accuracy = 100. * equal_targets / len(targets)
        return accuracy

    def get_loss(self, target_scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets: ground-truth targets of size [batch_size, max_target_length]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        targets = self.remove_start_of_sequence(targets)

        # Calculate the loss.
        _, _, vocabulary_size = target_scores.size()
        target_scores_2d = target_scores.reshape(-1, vocabulary_size)
        loss = self.loss_criterion(target_scores_2d, targets.view(-1))
        return loss

    def get_auxiliary_loss(self, auxiliary_scores_target: torch.Tensor, target_target_positions: torch.Tensor):
        target_loss = self.auxiliary_loss_criterion(auxiliary_scores_target, target_target_positions.view(-1))
        return target_loss

    def auxiliary_task_forward(self, output_scores_target_pos: torch.Tensor) -> torch.Tensor:
        assert self.auxiliary_task, "Please set auxiliary_task to True if using it."
        batch_size, _ = output_scores_target_pos.size()
        output_scores_target_pos = F.log_softmax(output_scores_target_pos, -1)
        return output_scores_target_pos

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        hidden, encoder_outputs = self.encoder(commands_input, commands_lengths)
        return {"encoded_situations": encoded_image, "encoded_commands": encoder_outputs, "hidden_states": hidden}

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor, target_lengths, curr_position) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations, target_lengths = target_lengths,
                                                   curr_position = curr_position)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths,
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched, _, context_situation = self.attention_decoder(input_tokens=target_batch,
                                                                              input_lengths=target_lengths,
                                                                              init_hidden=initial_hidden,
                                                                              encoded_commands=encoded_commands,
                                                                              commands_lengths=command_lengths,
                                                                              encoded_situations=encoded_situations,
                                                                              )
        decoder_output_batched = F.log_softmax(decoder_output_batched, dim=-1)
        length_penalty = (target_lengths - 2 - torch.arange(decoder_output_batched.size(0))).unsqueeze(1).to(decoder_output_batched)   ## length, 1,
        decoder_output_batched[:target_lengths - 2,:, self.target_eos_idx] = F.relu(decoder_output_batched[:target_lengths - 2,:, self.target_eos_idx] - self.penalty_alpha * F.relu(length_penalty))
        return decoder_output_batched, context_situation

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                           situations_input=situations_input)
        decoder_output, context_situation = self.decode_input_batched(
            target_batch=target_batch, target_lengths=target_lengths, initial_hidden=encoder_output["hidden_states"],
            encoded_commands=encoder_output["encoded_commands"]["encoder_outputs"], command_lengths=commands_lengths,
            encoded_situations=encoder_output["encoded_situations"])
        if self.auxiliary_task:
            target_position_scores = self.auxiliary_task_forward(context_situation)
        else:
            target_position_scores = torch.zeros(1), torch.zeros(1)
        return (decoder_output.transpose(0, 1),  # [batch_size, max_target_seq_length, target_vocabulary_size]
                target_position_scores)

    def update_state(self, is_best: bool, accuracy=None, exact_match=None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.best_exact_match = exact_match
            self.best_accuracy = accuracy
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings = 100, hidden_dropout_prob = 0.1, layer_norm_eps = 1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx= pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps = layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids = None, token_type_ids = None, inputs_embeds = None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, 0 : seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings

        embeddings = embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PredNetTransformer(nn.Module):
    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, num_transformer_layers:int, **kwargs):
        super(PredNetTransformer, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = cnn_hidden_num_channels * 3, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 4 * encoder_hidden_size, intermediate_size2 = cnn_hidden_num_channels * 3 * 4, num_attention_heads = 4, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.hidden_to_output = nn.Linear(encoder_hidden_size + cnn_hidden_num_channels * 3, 1, bias=False)


        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.MSELoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            if reduction == 'mean':
                gap = torch.abs(pred_lengths - target_lengths).mean()
            else:
                gap = torch.abs(pred_lengths - target_lengths)
        return gap

    def get_loss(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        if reduction == 'mean':
            loss = self.loss_criterion(pred_lengths, target_lengths)
        else:
            loss = F.mse_loss(pred_lengths, target_lengths, reduction = 'none')
        return loss

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        text_embeddings = self.text_embedding(commands_input)
        return encoded_image, text_embeddings

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched = self.attention_decoder(input_tokens=target_batch,
                                                                        input_lengths=target_lengths,
                                                                        init_hidden=initial_hidden,
                                                                        encoded_commands=encoded_commands,
                                                                        commands_lengths=command_lengths,
                                                                        encoded_situations=encoded_situations)
        return decoder_output_batched

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                                             situations_input=situations_input)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        length_pred = self.hidden_to_output(hidden).view(-1)

        return length_pred

    def update_state(self, is_best: bool, l2=None, l1=None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.l2 = l2
            self.l1 = l1
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

class PredNetTransformerV2(nn.Module):
    def __init__(self, input_vocabulary_size: int, embedding_dimension: int, encoder_hidden_size: int,
                 num_encoder_layers: int, target_vocabulary_size: int, encoder_dropout_p: float,
                 encoder_bidirectional: bool, num_decoder_layers: int, decoder_dropout_p: float,
                 decoder_hidden_size: int, num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, conditional_attention: bool, auxiliary_task: bool,
                 simple_situation_representation: bool, attention_type: str, num_transformer_layers:int, **kwargs):
        super(PredNetTransformerV2, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 1, bias=False)


        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.MSELoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            if reduction == 'mean':
                gap = torch.abs(pred_lengths - target_lengths).mean()
            else:
                gap = torch.abs(pred_lengths - target_lengths)
        return gap

    def get_loss(self, pred_lengths: torch.Tensor, target_lengths: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        if reduction == 'mean':
            loss = self.loss_criterion(pred_lengths, target_lengths)
        else:
            loss = F.mse_loss(pred_lengths, target_lengths, reduction = 'none')
        return loss

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input)
        return encoded_image, text_embeddings

    def decode_input(self, target_token: torch.LongTensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoder_outputs: torch.Tensor, input_lengths: List[int],
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step based on the previous hidden state of the decoder and the previous target token."""
        return self.attention_decoder.forward_step(input_tokens=target_token, last_hidden=hidden,
                                                   encoded_commands=encoder_outputs, commands_lengths=input_lengths,
                                                   encoded_situations=encoded_situations)

    def decode_input_batched(self, target_batch: torch.LongTensor, target_lengths: List[int],
                             initial_hidden: torch.Tensor, encoded_commands: torch.Tensor,
                             command_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor,
                                                                                                    torch.Tensor]:
        """Decode a batch of input sequences."""
        initial_hidden = self.attention_decoder.initialize_hidden(
            self.tanh(self.enc_hidden_to_dec_hidden(initial_hidden)))
        decoder_output_batched = self.attention_decoder(input_tokens=target_batch,
                                                                        input_lengths=target_lengths,
                                                                        init_hidden=initial_hidden,
                                                                        encoded_commands=encoded_commands,
                                                                        commands_lengths=command_lengths,
                                                                        encoded_situations=encoded_situations)
        return decoder_output_batched

    def forward(self, commands_input: torch.LongTensor, commands_lengths: List[int], situations_input: torch.Tensor,
                target_batch: torch.LongTensor, target_lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                                             situations_input=situations_input)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        length_pred = self.hidden_to_output(hidden).view(-1)

        return length_pred

    def update_state(self, is_best: bool, l2=None, l1=None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.l2 = l2
            self.l1 = l1
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

class Discriminator(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, **kwargs):
        super(Discriminator, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 2, bias=True)


        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label)
        precision = precision_score(labels, pred_label)
        F1 = f1_score(labels, pred_label)

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor, commands_lengths: List[int],
                     situations_input: torch.Tensor, token_type_ids) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input, token_type_ids = token_type_ids)
        return encoded_image, text_embeddings


    def forward(self, commands_input: torch.LongTensor, token_type_ids, commands_lengths: List[int], situations_input: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input, commands_lengths=commands_lengths,
                                                             situations_input=situations_input, token_type_ids = token_type_ids)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        logits = self.hidden_to_output(hidden)

        return logits

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

class DiscriminatorMultipleChoice(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, **kwargs):
        super(DiscriminatorMultipleChoice, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor,
                     situations_input: torch.Tensor, token_type_ids) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input, token_type_ids = token_type_ids)
        return encoded_image, text_embeddings


    def forward(self, commands_input: torch.LongTensor, token_type_ids, commands_lengths: List[int], situations_input: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, choice_num, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, token_type_ids = token_type_ids)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        _, image_size, image_hidden_size = encoded_image.size()
        encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        # encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1).view(batch_size * choice_num, image_size, image_hidden_size)
        # print("image hidden: ", encoded_image.size())
        # print("text embedding: ", text_embeddings.size())
        encoded_image = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size)
        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        hidden = hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        return logits

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class DiscriminatorMCAdv(DiscriminatorMultipleChoice):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, **kwargs):

        super().__init__(input_vocabulary_size, encoder_hidden_size, num_cnn_channels, cnn_kernel_size, cnn_dropout_p, 
                        cnn_hidden_num_channels, input_padding_idx, target_pad_idx,
                        target_eos_idx, output_directory, 
                        simple_situation_representation, num_transformer_layers, **kwargs)
        
    
    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction='mean') -> torch.Tensor:
        return super().get_loss(logits, labels, reduction)

    def forward(self, commands_input: torch.LongTensor, token_type_ids, situations_input: torch.Tensor, pred_length = True,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, choice_num, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, token_type_ids = token_type_ids)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        _, image_size, image_hidden_size = encoded_image.size()
        encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        # encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1).view(batch_size * choice_num, image_size, image_hidden_size)
        # print("image hidden: ", encoded_image.size())
        # print("text embedding: ", text_embeddings.size())
        encoded_image = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size)
        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        pooled_text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([pooled_text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        hidden = hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)
        
        outputs = (logits,)
        if pred_length:
            choice_mask = token_type_ids.eq(1).view(batch_size * choice_num, seq_len, 1).float()
            choice_hidden = text_hidden * choice_mask
            choice_hidden = torch.mean(choice_hidden, dim = 1).view(batch_size, choice_num, -1)
            outputs += (choice_hidden, )

        return outputs

class DiscOfDisc(nn.Module):
    def __init__(self, encoder_hidden_size):
        super().__init__()
        self.length_pred_head = nn.Linear(encoder_hidden_size, 1)
    
    def forward(self, choice_hidden):
        pred_length = self.length_pred_head(choice_hidden).squeeze(-1)
        return pred_length

    def get_loss(self, pred_lengths, target_lengths):
        batch_size, choice_num = pred_lengths.size()
        pred_lengths = pred_lengths.view(-1)
        target_lengths = target_lengths.view(-1)
        # loss = F.mse_loss(pred_lengths, target_lengths, reduction = 'mean')
        loss = F.smooth_l1_loss(pred_lengths, target_lengths, reduction = 'mean')
        return loss

    def get_metrics(self, pred_lengths, target_lengths):
        gap = torch.abs(pred_lengths - target_lengths).mean()
        return gap

class DiscOfDiscCls(nn.Module):
    def __init__(self, encoder_hidden_size, class_num = 20):
        super().__init__()
        self.length_cls_head = nn.Linear(encoder_hidden_size, class_num)
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, choice_hidden):
        pred_length = self.length_cls_head(choice_hidden)
        return pred_length

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, ) -> torch.Tensor:
        batch_size, choice_num, num_labels = logits.size()
        logits = logits.view(-1, num_labels)
        labels = labels.view(-1)
        loss = self.loss_criterion(logits, labels)
        return loss

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        labels = labels.view(-1)
        batch_size, choice_num, num_labels = logits.size()
        logits = logits.view(-1, num_labels)

        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        return acc

class DiscOfDiscCls_MLP(nn.Module):
    def __init__(self, encoder_hidden_size, intermediate_size_list = [512, 256],class_num = 20, dropout = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(encoder_hidden_size, intermediate_size_list[0])
        self.layer2 = nn.Linear(intermediate_size_list[0], intermediate_size_list[1])


        self.length_cls_head = nn.Linear(intermediate_size_list[-1], class_num)

        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = dropout)


        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, choice_hidden):
        layer1_out = self.layer1(choice_hidden)
        layer1_out = self.act(layer1_out)
        layer1_out = self.dropout(layer1_out)

        layer2_out = self.layer2(layer1_out)
        layer2_out = self.act(layer2_out)
        layer2_out = self.dropout(layer2_out)

        pred_length = self.length_cls_head(layer2_out)
        return pred_length

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, ) -> torch.Tensor:
        batch_size, choice_num, num_labels = logits.size()
        logits = logits.view(-1, num_labels)
        labels = labels.view(-1)
        loss = self.loss_criterion(logits, labels)
        return loss

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        labels = labels.view(-1)
        batch_size, choice_num, num_labels = logits.size()
        logits = logits.view(-1, num_labels)

        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        return acc


class DiscriminatorMC_SimpleFeat(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, **kwargs):
        super(DiscriminatorMC_SimpleFeat, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        transformer_layer = TransformerEncoderLayer(d_model = encoder_hidden_size,nhead = 2, dim_feedforward = 2 * encoder_hidden_size,
                                    dropout = 0.1, activation = 'relu', 
                                    )
        self.text_encoder = TransformerEncoder(encoder_layer = transformer_layer, num_layers = self.num_transformer_layers)

        
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)

        self.choice_vec_proj = nn.Linear(5, encoder_hidden_size)

        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor,
                     situations_input: torch.Tensor,) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input, )
        return encoded_image, text_embeddings


    def forward(self, commands_input: torch.LongTensor, situations_input: torch.Tensor,
                choice_vectors, 
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, )
        choice_num = choice_vectors.size(1)
        choice_hidden = self.choice_vec_proj(choice_vectors).view(batch_size * choice_num, 1, -1)   ## batch_size, choice_num, hidden_dim
        text_embeddings = text_embeddings.view(batch_size,1, seq_len, -1).expand(-1, choice_num, -1, -1).reshape(batch_size * choice_num, seq_len, -1)
        
        text_embeddings = choice_hidden + text_embeddings

        commands_input = commands_input.view(batch_size,1, seq_len).expand(-1, choice_num, -1).reshape(batch_size * choice_num, seq_len)
        attention_mask_text = commands_input.eq(self.input_padding_idx)
        text_embeddings = text_embeddings.transpose(0,1)
        encoded_text = self.text_encoder(src = text_embeddings, src_key_padding_mask = attention_mask_text) ## encoder_out: seq_len, batch_size * choice_num, hidden_size
        encoded_text = encoded_text.transpose(0,1)



        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        _, image_size, image_hidden_size = encoded_image.size()
        encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        encoded_image = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size)

        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](encoded_text, attention_mask_text, encoded_image, None, None)
            encoded_text, encoded_image = text_hidden, image_hidden
        pooled_text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([pooled_text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        hidden = hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        return logits

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class DiscriminatorMC_SimpleFeatV2(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, **kwargs):
        super(DiscriminatorMC_SimpleFeatV2, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = 0.1, dropout2 = 0.1, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)

        self.choice_vec_proj = nn.Linear(5, encoder_hidden_size)

        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor,
                     situations_input: torch.Tensor,) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input, )
        return encoded_image, text_embeddings


    def forward(self, commands_input: torch.LongTensor, situations_input: torch.Tensor,
                choice_vectors, 
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, )
        choice_num = choice_vectors.size(1)
        choice_hidden = self.choice_vec_proj(choice_vectors).view(batch_size * choice_num, 1, -1)   ## batch_size, choice_num, hidden_dim
        text_embeddings = text_embeddings.view(batch_size,1, seq_len, -1).expand(-1, choice_num, -1, -1).reshape(batch_size * choice_num, seq_len, -1)
        

        text_embeddings = torch.cat([choice_hidden, text_embeddings], dim = 1)

        commands_input = commands_input.view(batch_size,1, seq_len).expand(-1, choice_num, -1).reshape(batch_size * choice_num, seq_len)
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        choice_mask = torch.ones((batch_size * choice_num, 1), device = text_embeddings.device).bool()
        attention_mask_text = torch.cat([choice_mask, attention_mask_text], dim = 1)
        _, image_size, image_hidden_size = encoded_image.size()
        encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        # encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1).view(batch_size * choice_num, image_size, image_hidden_size)
        # print("image hidden: ", encoded_image.size())
        # print("text embedding: ", text_embeddings.size())
        encoded_image = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size)


        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](text_embeddings, attention_mask_text, encoded_image, None, None)
            text_embeddings, encoded_image = text_hidden, image_hidden
        text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)

        ## choice_vectors:  batch_size, choice_num, 5
        
        hidden = torch.cat([text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        return logits

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class DiscTextOnly(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 num_transformer_layers:int, **kwargs):
        super(DiscTextOnly, self).__init__()
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        self.num_transformer_layers = num_transformer_layers

        transformer_layer = TransformerEncoderLayer(d_model = encoder_hidden_size,nhead = 2, dim_feedforward = 2 * encoder_hidden_size,
                                            dropout = 0.1, activation = 'relu', 
                                            )

        self.encoder = TransformerEncoder(encoder_layer = transformer_layer, num_layers = self.num_transformer_layers)
    
        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.hidden_to_output = nn.Linear(encoder_hidden_size, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor, token_type_ids) -> Dict[str, torch.Tensor]:
        text_embeddings = self.text_embedding(commands_input, token_type_ids = token_type_ids)
        return text_embeddings

    def forward(self, commands_input: torch.LongTensor, token_type_ids,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, choice_num, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)
        text_embeddings = self.encode_input(commands_input=commands_input, token_type_ids = token_type_ids)
        
        text_embeddings = text_embeddings.transpose(0,1)

        attention_mask_text = commands_input.eq(self.input_padding_idx)
        encoder_out = self.encoder(src = text_embeddings, src_key_padding_mask = attention_mask_text)

        text_hidden = encoder_out[0,:]
        
        hidden = text_hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        return logits

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path

class DiscriminatorMC_V2(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, dropout:float, **kwargs):
        super(DiscriminatorMC_V2, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        dropout = dropout
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx, hidden_dropout_prob = dropout)
        cross_modal_transformer_layer = BiModalConnectionLayer(hidden_size1 = encoder_hidden_size, hidden_size2 = encoder_hidden_size, combined_hidden_size = encoder_hidden_size, 
                                                                intermediate_size1 = 2 * encoder_hidden_size, intermediate_size2 = 2 * encoder_hidden_size, num_attention_heads = 8, 
                                                                dropout1 = dropout, dropout2 = dropout, activation = 'relu')
        self.num_transformer_layers = num_transformer_layers
        self.cross_modal_transformer = replicate_layers(cross_modal_transformer_layer, num_copies = num_transformer_layers)

        transformer_layer = TransformerEncoderLayer(d_model = encoder_hidden_size,nhead = 2, dim_feedforward = 2 * encoder_hidden_size,
                                    dropout = dropout, activation = 'relu', 
                                    )

        self.text_encoder = TransformerEncoder(encoder_layer = transformer_layer, num_layers = self.num_transformer_layers)


        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size * 2, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor,
                     situations_input: torch.Tensor, token_type_ids) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        text_embeddings = self.text_embedding(commands_input, token_type_ids = token_type_ids)
        return encoded_image, text_embeddings

    def forward(self, commands_input: torch.LongTensor, token_type_ids,  situations_input: torch.Tensor,
                pred_length = True, labels = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, choice_num, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, token_type_ids = token_type_ids)
        
        attention_mask_text = commands_input.eq(self.input_padding_idx)
        text_embeddings = text_embeddings.transpose(0,1)
        encoded_text = self.text_encoder(src = text_embeddings, src_key_padding_mask = attention_mask_text) ## encoder_out: seq_len, batch_size * choice_num, hidden_size
        encoded_text = encoded_text.transpose(0,1)
        
        attention_mask_text = ~commands_input.eq(self.input_padding_idx)
        _, image_size, image_hidden_size = encoded_image.size()
        # encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        # encoded_image = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size)

        encoded_image = encoded_image.unsqueeze(1).repeat(1, choice_num, 1, 1)
        encoded_image = encoded_image.view(batch_size * choice_num, image_size, image_hidden_size)


        for idx in range(len(self.cross_modal_transformer)):
            text_hidden, image_hidden = self.cross_modal_transformer[idx](encoded_text, attention_mask_text, encoded_image, None, None)
            encoded_text, encoded_image = text_hidden, image_hidden
        pooled_text_hidden = text_hidden[:,0]
        image_hidden = torch.mean(image_hidden, dim = 1)
        hidden = torch.cat([pooled_text_hidden, image_hidden], dim = -1)  ## batch_size, text_hidden + image_hidden
        
        hidden = hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        outputs = (logits,)
        if pred_length:
            # choice_mask = token_type_ids.eq(1).view(batch_size * choice_num, seq_len, 1).float()
            # choice_hidden = text_hidden * choice_mask
            # choice_hidden = torch.mean(choice_hidden, dim = 1).view(batch_size, choice_num, -1)
            # outputs += (choice_hidden, )
            outputs += (hidden, )

        if labels != None:
            loss = self.get_loss(logits, labels)
            outputs += (loss,)
        return outputs

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path


class DiscriminatorMC_V3(nn.Module):
    def __init__(self, input_vocabulary_size: int, encoder_hidden_size: int,
                 num_cnn_channels: int, cnn_kernel_size: int,
                 cnn_dropout_p: float, cnn_hidden_num_channels: int, input_padding_idx: int, target_pad_idx: int,
                 target_eos_idx: int, output_directory: str, 
                 simple_situation_representation: bool, num_transformer_layers:int, 
                 image_size = 12 * 12,
                 **kwargs):
        super(DiscriminatorMC_V3, self).__init__()

        self.simple_situation_representation = simple_situation_representation
        if not simple_situation_representation:
            logger.warning("DownSamplingConvolutionalNet not correctly implemented. Update or set "
                           "--simple_situation_representation")
            self.downsample_image = DownSamplingConvolutionalNet(num_channels=num_cnn_channels,
                                                                 num_conv_channels=cnn_hidden_num_channels,
                                                                 dropout_probability=cnn_dropout_p)
            cnn_input_channels = cnn_hidden_num_channels
        else:
            cnn_input_channels = num_cnn_channels
        # Input: [batch_size, image_width, image_width, num_channels]
        # Output: [batch_size, image_width * image_width, num_conv_channels * 3]
        self.situation_encoder = ConvolutionalNet(num_channels=cnn_input_channels,
                                                  cnn_kernel_size=cnn_kernel_size,
                                                  num_conv_channels=cnn_hidden_num_channels,
                                                  dropout_probability=cnn_dropout_p)
        self.text_embedding = TransformerEmbedding(vocab_size = input_vocabulary_size, hidden_size = encoder_hidden_size, pad_token_id = input_padding_idx,)
        self.num_transformer_layers = num_transformer_layers
        transformer_layer = TransformerEncoderLayer(d_model = encoder_hidden_size,nhead = 2, dim_feedforward = 2 * encoder_hidden_size,
                                    dropout = 0.1, activation = 'relu', 
                                    )

        self.transformer_encoder = TransformerEncoder(encoder_layer = transformer_layer, num_layers = self.num_transformer_layers)

        self.image_positional_embedding = nn.Parameter(torch.zeros(1, image_size, encoder_hidden_size))

        self.dropout = nn.Dropout(p = 0.1)


        # Input: [batch_size, max_input_length]
        # Output: [batch_size, hidden_size], [batch_size, max_input_length, hidden_size]
        self.cnn_proj = nn.Linear(cnn_hidden_num_channels * 3, encoder_hidden_size)
        self.hidden_to_output = nn.Linear(encoder_hidden_size, 1, bias=True)

        self.encoder_hidden_size = encoder_hidden_size
        self.target_eos_idx = target_eos_idx
        self.target_pad_idx = target_pad_idx
        self.input_padding_idx = input_padding_idx
        self.loss_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        self.tanh = nn.Tanh()
        self.output_directory = output_directory
        self.trained_iterations = 0
        self.best_iteration = 0
        self.best_exact_match = 0
        self.best_accuracy = 0

    def get_metrics(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> Tuple[float, float]:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size, max_target_length, target_vocab_size]
        :param targets:  ground-truth targets of size [batch_size, max_target_length]
        :return: scalar float of accuracy averaged over sequence length and batch size.
        """
        with torch.no_grad():
            loss = self.loss_criterion(logits, labels)
        pred_label = torch.argmax(logits, dim = 1)
        acc = (pred_label == labels).sum().item() / logits.size(0)

        pred_label = pred_label.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        recall = recall_score(labels, pred_label, average = 'micro')
        precision = precision_score(labels, pred_label, average = 'micro')
        F1 = f1_score(labels, pred_label, average = 'micro')

        return loss, acc, recall, precision, F1

    def get_loss(self, logits: torch.Tensor, labels: torch.Tensor, reduction = 'mean') -> torch.Tensor:
        """
        :param target_scores: probabilities over target vocabulary outputted by the model, of size
                              [batch_size]
        :param targets: ground-truth targets of size [batch_size]
        :return: scalar negative log-likelihood loss averaged over the sequence length and batch size.
        """
        loss = self.loss_criterion(logits, labels)
        return loss

    def encode_input(self, commands_input: torch.LongTensor,
                     situations_input: torch.Tensor, token_type_ids) -> Dict[str, torch.Tensor]:
        """Pass the input commands through an RNN encoder and the situation input through a CNN encoder."""
        if not self.simple_situation_representation:
            situations_input = self.downsample_image(situations_input)
        encoded_image = self.situation_encoder(situations_input)
        encoded_image = self.cnn_proj(encoded_image)
        encoded_image  = encoded_image + self.image_positional_embedding
        
        text_embeddings = self.text_embedding(commands_input, token_type_ids = token_type_ids)

        encoded_image = self.dropout(encoded_image)
        text_embeddings = self.dropout(text_embeddings)

        return encoded_image, text_embeddings

    def forward(self, commands_input: torch.LongTensor, token_type_ids,  situations_input: torch.Tensor,
                pred_length = True,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## commands_input:  batch_size, choice_num, seq_len
        batch_size, choice_num, seq_len = commands_input.size()
        commands_input = commands_input.view(-1, seq_len)
        token_type_ids = token_type_ids.view(-1, seq_len)
        encoded_image, text_embeddings = self.encode_input(commands_input=commands_input,
                                                             situations_input=situations_input, token_type_ids = token_type_ids)


        attention_mask_text = commands_input.eq(self.input_padding_idx)
        attention_mask_image = torch.zeros((batch_size * choice_num, encoded_image.size(1)), device = device).bool()
        input_mask = torch.cat([attention_mask_text, attention_mask_image], dim = 1)

        _, image_size, image_hidden_size = encoded_image.size()
        encoded_image = encoded_image.unsqueeze(1).expand(-1, choice_num, -1, -1)
        image_embeddings = encoded_image.reshape(batch_size * choice_num, image_size, image_hidden_size).transpose(0, 1)
        text_embeddings = text_embeddings.transpose(0, 1)
        input_embeddings = torch.cat([text_embeddings, image_embeddings], dim = 0)   ## seq_len,  batch_size,  hidden_size

        ## size check
        # print(f"batch size: {batch_size}, choice num: {choice_num}, seq len: {seq_len}, image size: {image_size}")
        # print("text embeddings: ", text_embeddings.size())
        # print("text mask: ", attention_mask_text.size())
        # print("image embeddings: ", image_embeddings.size())
        # print("concat input: ", input_embeddings.size())
        # print("concat mask: ", input_mask.size())
        # pause = input("???")
        ## sizhe check end

        encoded_hidden = self.transformer_encoder(src = input_embeddings, src_key_padding_mask = input_mask) ## encoder_out: seq_len, batch_size * choice_num, hidden_size
        encoded_hidden = encoded_hidden.transpose(0,1)   ## batch_size * choice_num, seq_len,  hidden_size
        
        pooled_hidden = encoded_hidden[:,0]
        
        hidden = pooled_hidden.view(batch_size, choice_num, -1)

        logits = self.hidden_to_output(hidden).view(batch_size, choice_num)

        outputs = (logits,)
        if pred_length:
            # choice_mask = token_type_ids.eq(1).view(batch_size * choice_num, seq_len, 1).float()
            # choice_hidden = text_hidden * choice_mask
            # choice_hidden = torch.mean(choice_hidden, dim = 1).view(batch_size, choice_num, -1)
            outputs += (hidden, )

        return outputs

    def update_state(self, is_best: bool, acc=None, F1=None, recall = None, precision = None) -> dict():
        self.trained_iterations += 1
        if is_best:
            self.acc = acc
            self.F1 = F1
            self.recall = recall
            self.precision = precision
            self.best_iteration = self.trained_iterations

    def load_model(self, path_to_checkpoint: str) -> dict:
        checkpoint = torch.load(path_to_checkpoint)
        self.trained_iterations = checkpoint["iteration"]
        self.best_iteration = checkpoint["best_iteration"]
        self.load_state_dict(checkpoint["state_dict"])
        self.best_exact_match = checkpoint["best_exact_match"]
        self.best_accuracy = checkpoint["best_accuracy"]
        return checkpoint["optimizer_state_dict"]

    def get_current_state(self):
        return {
            "iteration": self.trained_iterations,
            "state_dict": self.state_dict(),
            "best_iteration": self.best_iteration,
            "best_accuracy": self.best_accuracy,
            "best_exact_match": self.best_exact_match
        }

    def save_checkpoint(self, file_name: str, is_best: bool, optimizer_state_dict: dict) -> str:
        """

        :param file_name: filename to save checkpoint in.
        :param is_best: boolean describing whether or not the current state is the best the model has ever been.
        :param optimizer_state_dict: state of the optimizer.
        :return: str to path where the model is saved.
        """
        path = os.path.join(self.output_directory, file_name)
        state = self.get_current_state()
        state["optimizer_state_dict"] = optimizer_state_dict
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.output_directory, 'model_best.pth.tar')
            shutil.copyfile(path, best_path)
        return path



