import numpy as np
import pytest
from torch import equal, long, tensor

from language_modeling.domain.modeling import DEVICE
from language_modeling.domain.modeling.utils.data.dataloader import (
    LanguageModelingDataLoader,
)


def test_language_modeling_dataloader_check_if_sequence_of_ids_is_a_tuple_should_return_true_when_sequence_of_ids_is_a_tuple():
    # Given
    sequence_of_ids = tuple((1, 2))
    language_modeling_dataloader = LanguageModelingDataLoader(1, sequence_of_ids)

    # When
    output = language_modeling_dataloader._check_if_sequence_of_ids_is_a_tuple(
        sequence_of_ids
    )

    # Then
    assert output


def test_language_modeling_dataloader_check_if_sequence_of_ids_is_a_tuple_should_return_false_when_sequence_of_ids_is_not_a_tuple():
    # Given
    sequence_of_ids = 1
    language_modeling_dataloader = LanguageModelingDataLoader(1, sequence_of_ids)

    # When
    output = language_modeling_dataloader._check_if_sequence_of_ids_is_a_tuple(
        sequence_of_ids
    )

    # Then
    assert not output


def test_language_modeling_dataloader_check_sequence_of_ids_length_should_raise_exception_for_a_3_element_tuple():
    # Given
    sequence_of_ids = tuple((1, 2, 3))
    language_modeling_dataloader = LanguageModelingDataLoader(1, sequence_of_ids)

    # When
    with pytest.raises(ValueError) as exc_info:
        _ = language_modeling_dataloader._check_if_sequence_of_ids_is_a_tuple(
            sequence_of_ids
        )

    # Then
    assert exc_info.type == ValueError


def test_language_modeling_dataloader_get_batches_should_return_a_tuple_of_tensor_when_sequence_of_ids_is_a_tuple():
    # Given
    batch_index = 1
    bptt = 2
    train_sequence_of_ids = np.zeros((5, 10))
    target_sequence_of_ids = np.zeros((5, 10))
    sequence_of_ids = (train_sequence_of_ids, target_sequence_of_ids)
    language_modeling_dataloader = LanguageModelingDataLoader(bptt, sequence_of_ids)

    # When
    output = language_modeling_dataloader.get_batches(batch_index)
    output = next(output)

    # Then
    assert equal(
        output[0],
        tensor(
            train_sequence_of_ids[:, batch_index : batch_index + bptt],
            device=DEVICE,
            dtype=long,
        ),
    )
    assert equal(
        output[1],
        tensor(
            target_sequence_of_ids[:, batch_index : batch_index + bptt],
            device=DEVICE,
            dtype=long,
        ),
    )


def test_language_modeling_dataloader_get_batches_should_return_a_tensor_when_sequence_of_ids_is_a_ndarray():
    # Given
    batch_index = 1
    bptt = 2
    train_sequence_of_ids = np.zeros((5, 10))
    sequence_of_ids = train_sequence_of_ids
    language_modeling_dataloader = LanguageModelingDataLoader(bptt, sequence_of_ids)

    # When
    output = language_modeling_dataloader.get_batches(batch_index)

    # Then
    assert equal(
        next(output),
        tensor(
            sequence_of_ids[:, batch_index : batch_index + bptt],
            device=DEVICE,
            dtype=long,
        ),
    )


def test_language_modeling_dataloader_len_should_correct_output_when_sequence_of_ids_is_a_tuple():
    # Given
    bptt = 2
    train_sequence_of_ids = np.zeros((5, 10))
    target_sequence_of_ids = np.zeros((5, 10))
    sequence_of_ids = (train_sequence_of_ids, target_sequence_of_ids)
    language_modeling_dataloader = LanguageModelingDataLoader(bptt, sequence_of_ids)
    expected = 10

    # When
    output = len(language_modeling_dataloader)

    # Then
    assert output == expected


def test_language_modeling_dataloader_len_should_correct_output_when_sequence_of_ids_is_not_a_tuple():
    # Given
    bptt = 2
    sequence_of_ids = np.zeros((5, 10))
    language_modeling_dataloader = LanguageModelingDataLoader(bptt, sequence_of_ids)
    expected = 10

    # When
    output = len(language_modeling_dataloader)

    # Then
    assert output == expected
