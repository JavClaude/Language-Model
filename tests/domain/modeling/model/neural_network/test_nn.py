from unittest.mock import patch, MagicMock

import torch

from language_modeling.domain.modeling.model.neural_network.nn import (
    EmbeddingBlock,
    ResBlock,
    DecoderBlock,
    LSTMModel,
)


@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.Embedding.forward",
    return_value=10,
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.LayerNorm.forward",
    return_value=5,
)
def test_embedding_block_should_call_embedding_and_layer_norm_layer(
    layernorm_mock, embedding_mock
):
    # Given
    inputs = MagicMock()
    embedding_block = EmbeddingBlock(10, 1)

    # When
    _ = embedding_block(inputs)

    # Then
    embedding_mock.assert_called_with(inputs)
    layernorm_mock.assert_called_with(10)


@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.Linear.forward",
    return_value=10,
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.ReLU.forward",
    return_value=5,
)
def test_res_block_should_linear_and_relu_layer(relu_mock, linear_mock):
    # Given
    inputs = MagicMock()
    res_block = ResBlock(10)

    # When
    _ = res_block(inputs)

    # Then
    linear_mock.assert_called_with(inputs)
    relu_mock.assert_called_with(10)


@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.LayerNorm.forward",
    return_value=10,
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.ResBlock.forward",
    return_value=5,
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.Linear.forward",
    return_value=5,
)
def test_decoder_block_should_linear_and_relu_layer(
    linear_mock, resblock_mock, layernorm_mock
):
    # Given
    inputs = MagicMock()
    n_blocks = 5
    decoder_block = DecoderBlock(10, 10, n_blocks)

    # When
    _ = decoder_block(inputs)

    # Then
    layernorm_mock.assert_called_with(inputs)
    assert resblock_mock.call_count == n_blocks
    linear_mock.assert_called_with(5)


@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.EmbeddingBlock.forward",
    return_value=10,
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.LSTM.forward",
    return_value=(5, 5),
)
@patch(
    "language_modeling.domain.modeling.model.neural_network.nn.DecoderBlock.forward",
    return_value=5,
)
def test_lstm_model_should_call_other_nn_blocks(
    decoder_block_mock, lstm_mock, embedding_block_mock
):
    # Given
    inputs = MagicMock()
    hidden_states = MagicMock()
    model = LSTMModel(100, 10, 10, 5, 1)

    # When
    _ = model(inputs, hidden_states)

    # Then
    embedding_block_mock.assert_called_with(inputs)
    lstm_mock.assert_called_with(10, hidden_states)
    decoder_block_mock.assert_called_with(5)


def test_lstm_model_init_hidden_states_should_return_correct_output():
    # Given
    expected = (torch.zeros(1, 64, 10), torch.zeros(1, 64, 10))
    model = LSTMModel(100, 10, 10, 5, 1)

    # When
    output = model.init_hidden_states(64)

    # Then
    assert torch.equal(output[0], expected[0])
    assert torch.equal(output[1], expected[1])
