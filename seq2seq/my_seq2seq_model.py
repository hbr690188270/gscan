import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import logging
from typing import List
from typing import Tuple

from seq2seq.helpers import sequence_mask

from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoderLayer, TransformerEncoder
from torch import Tensor, embedding
from typing import Optional, Any
from torch.nn import LayerNorm

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

class Attention(nn.Module):

    def __init__(self, key_size: int, query_size: int, hidden_size: int):
        super(Attention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, queries: torch.Tensor, projected_keys: torch.Tensor, values: torch.Tensor,
                memory_lengths: List[int]):
        """
        Key-value memory which takes queries and retrieves weighted combinations of values
          This version masks out certain memories, so that you can differing numbers of memories per batch.

        :param queries: [batch_size, 1, query_dim]
        :param projected_keys: [batch_size, num_memory, query_dim]
        :param values: [batch_size, num_memory, value_dim]
        :param memory_lengths: [batch_size] actual number of keys in each batch
        :return:
            soft_values_retrieval : soft-retrieval of values; [batch_size, 1, value_dim]
            attention_weights : soft-retrieval of values; [batch_size, 1, n_memory]
        """
        batch_size = projected_keys.size(0)
        assert len(memory_lengths) == batch_size
        memory_lengths = torch.tensor(memory_lengths, dtype=torch.long, device=device)

        # Project queries down to the correct dimension.
        # [bsz, 1, query_dimension] X [bsz, query_dimension, hidden_dim] = [bsz, 1, hidden_dim]
        queries = self.query_layer(queries)

        # [bsz, 1, query_dim] X [bsz, query_dim, num_memory] = [bsz, num_memory, 1]
        scores = self.energy_layer(torch.tanh(queries + projected_keys))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out keys that are on a padding location.encoded_commands
        mask = sequence_mask(memory_lengths)  # [batch_size, num_memory]
        mask = mask.unsqueeze(1)  # [batch_size, 1, num_memory]
        scores = scores.masked_fill(mask == 0, float('-inf'))  # fill with large negative numbers
        attention_weights = F.softmax(scores, dim=2)  # [batch_size, 1, num_memory]

        # [bsz, 1, num_memory] X [bsz, num_memory, value_dim] = [bsz, 1, value_dim]
        soft_values_retrieval = torch.bmm(attention_weights, values)
        return soft_values_retrieval, attention_weights

class LengthDecoder(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, textual_attention: Attention, num_layers,
                 visual_attention: Attention, dropout_probability=0.1,
                 conditional_attention=False):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(LengthDecoder, self).__init__()
        self.conditional_attention = conditional_attention
        if self.conditional_attention:
            self.queries_to_keys = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.sos_embedding = nn.parameter.Parameter(data = torch.Tensor(hidden_size))
        nn.init.normal_(self.sos_embedding)

        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_probability)
        self.textual_attention = textual_attention
        self.visual_attention = visual_attention
        self.output_to_hidden = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, 1, bias=False)

    def forward_step(self, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoded_commands: torch.Tensor, commands_lengths: torch.Tensor,
                     encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoded_commands: all encoder outputs, [max_input_length, batch_size, hidden_size]
        :param commands_lengths: length of each padded input seqencoded_commandsuence that were passed to the encoder.
        :param encoded_situations: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        last_hidden, last_cell = last_hidden
        batch_size, image_num_memory, _ = encoded_situations.size()

        # Embed each input symbol
        
        embed_input = self.sos_embedding.unsqueeze(0).expand(batch_size,-1)
        
        embed_input = self.dropout(embed_input)


        # Bahdanau attention
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), projected_keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context_command], dim=-1)
            queries = self.tanh(self.queries_to_keys(queries))
        else:
            queries = last_hidden.transpose(0, 1)

        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=encoded_situations,
            values=encoded_situations, memory_lengths=situation_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]

        context_command = context_command.transpose(0, 1).squeeze(0)
        context_situation = context_situation.transpose(0,1).squeeze(0)
        concat_input = torch.cat([embed_input,
                                  context_command,
                                  context_situation], dim=1)  # [1, batch_size hidden_size*3]

        concat_input = self.dropout(concat_input)

        pre_output = self.output_to_hidden(concat_input)  # [batch_size, hidden_size]

        pre_output = self.dropout(pre_output)
        output = self.hidden_to_output(pre_output)  # [batch_size, 1]
        output = output.squeeze(1)
        return output


    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoded_commands: torch.Tensor,
                commands_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, List[int],
                                                                                        torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [batch_size, image_width * image_width, image_features]; encoded image situations.
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        batch_size, seq_len = input_tokens.size()

        # Sort the sequences by length in descending order
        # initial_h, initial_c = init_hidden
        commands_lengths = torch.tensor(commands_lengths, device=device)

        # For efficiency
        projected_keys_visual = self.visual_attention.key_layer(
            encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        projected_keys_textual = self.textual_attention.key_layer(
            encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        output = self.forward_step(init_hidden, projected_keys_textual,
                                                            commands_lengths,
                                                            projected_keys_visual)

        # Reverse the sorting

        return output
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )

class AttentionDecoderWithLength(nn.Module):
    """One-step batch decoder with Luong et al. attention"""

    def __init__(self, hidden_size: int, output_size: int, num_layers: int, textual_attention: Attention,
                 visual_attention: Attention, dropout_probability=0.1, padding_idx=0,
                 conditional_attention=False, eos_idx = 2, alpha = 0.01):
        """
        :param hidden_size: number of hidden units in RNN, and embedding size for output symbols
        :param output_size: number of output symbols
        :param num_layers: number of hidden layers
        :param dropout_probability: dropout applied to symbol embeddings and RNNs
        """
        super(AttentionDecoderWithLength, self).__init__()
        self.num_layers = num_layers
        self.conditional_attention = conditional_attention
        if self.conditional_attention:
            self.queries_to_keys = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_probability = dropout_probability
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_probability)
        self.lstm = nn.LSTM(hidden_size * 3, hidden_size, num_layers=num_layers, dropout=dropout_probability)
        self.textual_attention = textual_attention
        self.visual_attention = visual_attention
        self.output_to_hidden = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size, bias=False)

        self.eos_idx = eos_idx


    def forward_step(self, input_tokens: torch.LongTensor, last_hidden: Tuple[torch.Tensor, torch.Tensor],
                     encoded_commands: torch.Tensor, commands_lengths: torch.Tensor,
                     encoded_situations: torch.Tensor, ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run batch decoder forward for a single time step.
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: one time step inputs tokens of length batch_size
        :param last_hidden: previous decoder state, which is pair of tensors [num_layers, batch_size, hidden_size]
        (pair for hidden and cell)
        :param encoded_commands: all encoder outputs, [max_input_length, batch_size, hidden_size]
        :param commands_lengths: length of each padded input seqencoded_commandsuence that were passed to the encoder.
        :param encoded_situations: the situation encoder outputs, [image_dimension * image_dimension, batch_size,
         hidden_size]
        :return: output : un-normalized output probabilities, [batch_size, output_size]
          hidden : current decoder state, which is a pair of tensors [num_layers, batch_size, hidden_size]
           (pair for hidden and cell)
          attention_weights : attention weights, [batch_size, 1, max_input_length]
        """
        last_hidden, last_cell = last_hidden

        # Embed each input symbol
        embedded_input = self.embedding(input_tokens)  # [batch_size, hidden_size]
        embedded_input = self.dropout(embedded_input)
        embedded_input = embedded_input.unsqueeze(0)  # [1, batch_size, hidden_size]

        # Bahdanau attention
        context_command, attention_weights_commands = self.textual_attention(
            queries=last_hidden.transpose(0, 1), projected_keys=encoded_commands.transpose(0, 1),
            values=encoded_commands.transpose(0, 1), memory_lengths=commands_lengths)
        batch_size, image_num_memory, _ = encoded_situations.size()
        situation_lengths = [image_num_memory for _ in range(batch_size)]

        if self.conditional_attention:
            queries = torch.cat([last_hidden.transpose(0, 1), context_command], dim=-1)
            queries = self.tanh(self.queries_to_keys(queries))
        else:
            queries = last_hidden.transpose(0, 1)

        context_situation, attention_weights_situations = self.visual_attention(
            queries=queries, projected_keys=encoded_situations,
            values=encoded_situations, memory_lengths=situation_lengths)
        # context : [batch_size, 1, hidden_size]
        # attention_weights : [batch_size, 1, max_input_length]

        # Concatenate the context vector and RNN hidden state, and map to an output
        attention_weights_commands = attention_weights_commands.squeeze(1)  # [batch_size, max_input_length]
        attention_weights_situations = attention_weights_situations.squeeze(1)  # [batch_size, im_dim * im_dim]
        concat_input = torch.cat([embedded_input,
                                  context_command.transpose(0, 1),
                                  context_situation.transpose(0, 1)], dim=2)  # [1, batch_size hidden_size*3]

        last_hidden = (last_hidden, last_cell)
        lstm_output, hidden = self.lstm(concat_input, last_hidden)
        # lstm_output: [1, batch_size, hidden_size]
        # hidden: tuple of each [num_layers, batch_size, hidden_size] (pair for hidden and cell)
        # output = self.hidden_to_output(lstm_output)  # [batch_size, output_size]
        # output = output.squeeze(dim=0)

        # Concatenate all outputs and project to output size.
        pre_output = torch.cat([embedded_input, lstm_output,
                                context_command.transpose(0, 1), context_situation.transpose(0, 1)], dim=2)
        pre_output = self.output_to_hidden(pre_output)  # [1, batch_size, hidden_size]
        output = self.hidden_to_output(pre_output)  # [batch_size, output_size]
        output = output.squeeze(dim=0)   # [batch_size, output_size]

        # length_penalty = F.relu(target_lengths -2 - curr_position)
        # output[:, self.eos_idx] = output[:, self.eos_idx] - self.alpha * length_penalty

        return (output, hidden, attention_weights_situations.squeeze(dim=1), attention_weights_commands,
                attention_weights_situations)
        # output : [un-normalized probabilities] [batch_size, output_size]
        # hidden: tuple of size [num_layers, batch_size, hidden_size] (for hidden and cell)
        # attention_weights: [batch_size, max_input_length]

    def forward(self, input_tokens: torch.LongTensor, input_lengths: List[int],
                init_hidden: Tuple[torch.Tensor, torch.Tensor], encoded_commands: torch.Tensor,
                commands_lengths: List[int], encoded_situations: torch.Tensor) -> Tuple[torch.Tensor, List[int],
                                                                                        torch.Tensor]:
        """
        Run batch attention decoder forward for a series of steps
         Each decoder step considers all of the encoder_outputs through attention.
         Attention retrieval is based on decoder hidden state (not cell state)

        :param input_tokens: [batch_size, max_length];  padded target sequences
        :param input_lengths: [batch_size] for sequence length of each padded target sequence
        :param init_hidden: tuple of tensors [num_layers, batch_size, hidden_size] (for hidden and cell)
        :param encoded_commands: [max_input_length, batch_size, embedding_dim]
        :param commands_lengths: [batch_size] sequence length of each encoder sequence (without padding)
        :param encoded_situations: [batch_size, image_width * image_width, image_features]; encoded image situations.
        :return: output : unnormalized log-score, [max_length, batch_size, output_size]
          hidden : current decoder state, tuple with each [num_layers, batch_size, hidden_size] (for hidden and cell)
        """
        batch_size, max_time = input_tokens.size()

        # Sort the sequences by length in descending order
        input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
        input_lengths, perm_idx = torch.sort(input_lengths, descending=True)
        input_tokens_sorted = input_tokens.index_select(dim=0, index=perm_idx)
        initial_h, initial_c = init_hidden
        hidden = (initial_h.index_select(dim=1, index=perm_idx),
                  initial_c.index_select(dim=1, index=perm_idx))
        encoded_commands = encoded_commands.index_select(dim=1, index=perm_idx)
        commands_lengths = torch.tensor(commands_lengths, device=device)
        commands_lengths = commands_lengths.index_select(dim=0, index=perm_idx)
        encoded_situations = encoded_situations.index_select(dim=0, index=perm_idx)

        # For efficiency
        projected_keys_visual = self.visual_attention.key_layer(
            encoded_situations)  # [batch_size, situation_length, dec_hidden_dim]
        projected_keys_textual = self.textual_attention.key_layer(
            encoded_commands)  # [max_input_length, batch_size, dec_hidden_dim]

        all_attention_weights = []
        lstm_output = []
        for time in range(max_time):
            input_token = input_tokens_sorted[:, time]
            (output, hidden, context_situation, attention_weights_commands,
             attention_weights_situations) = self.forward_step(input_token, hidden, projected_keys_textual,
                                                               commands_lengths,
                                                               projected_keys_visual)
            all_attention_weights.append(attention_weights_situations.unsqueeze(0))
            lstm_output.append(output.unsqueeze(0))
        lstm_output = torch.cat(lstm_output, dim=0)  # [max_time, batch_size, output_size]
        attention_weights = torch.cat(all_attention_weights, dim=0)  # [max_time, batch_size, situation_dim**2]

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = lstm_output.index_select(dim=1, index=unperm_idx)  # [max_time, batch_size, output_size]
        seq_len = input_lengths[unperm_idx].tolist()
        attention_weights = attention_weights.index_select(dim=1, index=unperm_idx)

        return lstm_output, seq_len, attention_weights.sum(dim=0)
        # output : [unnormalized log-score] [max_length, batch_size, output_size]
        # seq_len : length of each output sequence

    def initialize_hidden(self, encoder_message: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Populate the hidden variables with a message from the encoder.
        All layers, and both the hidden and cell vectors, are filled with the same message.
        :param encoder_message:  [batch_size, hidden_size] tensor
        :return: tuple of Tensors representing the hidden and cell state of shape: [num_layers, batch_size, hidden_dim]
        """
        encoder_message = encoder_message.unsqueeze(0)  # [1, batch_size, hidden_size]
        encoder_message = encoder_message.expand(self.num_layers, -1,
                                                 -1).contiguous()  # [num_layers, batch_size, hidden_size]
        return encoder_message.clone(), encoder_message.clone()

    def extra_repr(self) -> str:
        return "AttentionDecoderRNN\n num_layers={}\n hidden_size={}\n dropout={}\n num_output_symbols={}\n".format(
            self.num_layers, self.hidden_size, self.dropout_probability, self.output_size
        )


