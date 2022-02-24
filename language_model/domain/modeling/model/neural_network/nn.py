from typing import Tuple

from torch import tensor, zeros
from torch.cuda import is_available
from torch.nn import (
    Embedding,
    LayerNorm,
    Linear,
    LSTM,
    Module,
    ReLU,
    Sequential
)

from language_model.domain.modeling import CUDA, CPU

device = CUDA if is_available() else CPU


class EmbeddingBlock(Module):
    def __init__(self, vocabulary_size: int, embedding_dimension: int = 300) -> None:
        super(EmbeddingBlock, self).__init__()

        self.embedding_block = Sequential(
            Embedding(vocabulary_size, embedding_dimension),
            LayerNorm(embedding_dimension)
        )

    def forward(self, inputs: tensor) -> tensor:
        return self.embedding_block(inputs)


class ResBlock(Module):
    def __init__(self, hidden_units_from_lstm: int) -> None:
        super(ResBlock, self).__init__()

        self.resblock = Sequential(
            Linear(hidden_units_from_lstm, hidden_units_from_lstm),
            ReLU()
        )

    def forward(self, inputs: tensor) -> tensor:
        identity_input = inputs
        return self.resblock(inputs) + identity_input


class DecoderBlock(Module):
    def __init__(self, hidden_units_from_lstm: int, vocabulary_size: int, n_blocks: int = 5) -> None:
        super(DecoderBlock, self).__init__()

        self.decoder_block = Sequential(
            LayerNorm(hidden_units_from_lstm),
            *[
                ResBlock(hidden_units_from_lstm) for _ in range(n_blocks)
            ],
            Linear(hidden_units_from_lstm, vocabulary_size)
        )

    def forward(self, inputs: tensor) -> tensor:
        return self.decoder_block(inputs)


class LSTMModel(Module):
    def __init__(
            self,
            vocabulary_size: int,
            embedding_dimension: int,
            hidden_units_for_lstm: int,
            n_blocks_for_decoder: int,
            num_layers: int = 2,
            **kwargs
    ) -> None:
        super(LSTMModel, self).__init__()

        self._hidden_units = hidden_units_for_lstm
        self._num_lstm_layers = num_layers

        self.embedding_block = EmbeddingBlock(vocabulary_size, embedding_dimension)
        self.lstm_block = LSTM(
            embedding_dimension,
            hidden_units_for_lstm,
            batch_first=True,
            num_layers=num_layers
        )
        self.decoder_block = DecoderBlock(hidden_units_for_lstm, vocabulary_size, n_blocks_for_decoder)

    def forward(self, inputs: tensor, previous_hidden_states: tensor) -> Tuple[tensor, tensor]:
        inputs = self.embedding_block(inputs)
        inputs, hidden_states = self.lstm_block(inputs, previous_hidden_states)
        outputs = self.decoder_block(inputs)
        return outputs, hidden_states

    def init_hidden_states(self, batch_size: int) -> Tuple[tensor, tensor]:
        return (
            zeros(self._num_lstm_layers, batch_size, self._hidden_units, device=device),
            zeros(self._num_lstm_layers, batch_size, self._hidden_units, device=device)
        )
