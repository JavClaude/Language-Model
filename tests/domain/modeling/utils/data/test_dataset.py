from unittest.mock import patch, MagicMock

import pytest
from tokenizers.implementations import ByteLevelBPETokenizer

from language_modeling.domain.modeling.utils.data import (
    EOD_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    UNK_TOKEN,
    TOKENIZER_FIELD,
    CHECK_TOKENIZER_ERROR_MESSAGE,
    IS_FITTED_MESSAGE,
)
from language_modeling.domain.modeling.utils.data.dataset import LanguageModelingDataset
from language_modeling.domain.modeling.utils.data.errors import (
    MissingTokenizerError,
    NotFittedDatasetError,
)

FAKE_PATH_FOR_TEST = "fake/path/to/text"


def test_language_modeling_dataset_add_special_token_should_return_correct_pad_text():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    expected = f"{SOS_TOKEN} This is a test {EOD_TOKEN}"

    # When
    output = language_modeling_dataset._add_special_tokens("This is a test")

    # Then
    assert output == expected


def test_language_modeling_dataset_set_tokenizer_should_set_a_tokenizer_attribute():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)

    # When
    language_modeling_dataset.set_tokenizer("tokenizer")

    # Then
    assert hasattr(language_modeling_dataset, TOKENIZER_FIELD)


def test_language_modeling_dataset_check_tokenizer_should_raise_an_error_when_no_tokenizer_is_setup():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)

    # When
    with pytest.raises(MissingTokenizerError) as exc_info:
        language_modeling_dataset._check_tokenizer()

    # Then
    assert isinstance(exc_info.value, MissingTokenizerError)
    assert str(exc_info.value) == CHECK_TOKENIZER_ERROR_MESSAGE


def test_language_modeling_dataset_check_is_fitted_should_raise_an_error_when_the_object_is_not_fitted():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)

    # When
    with pytest.raises(NotFittedDatasetError) as exc_info:
        _ = language_modeling_dataset.transform(FAKE_PATH_FOR_TEST, True)

    # Then
    assert isinstance(exc_info.value, NotFittedDatasetError)
    assert str(exc_info.value) == IS_FITTED_MESSAGE


@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._check_tokenizer"
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._fit_tokenizer"
)
def test_language_modeling_dataset_fit_method_should_call_other_fitting_method(
    fit_tokenizer_mock, check_tokenizer_mock
):
    # Given
    fake_path_to_texts = "fake/path/to/text"
    vocabulary_size = 10
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    language_modeling_dataset.set_tokenizer("a")

    # When
    language_modeling_dataset.fit(FAKE_PATH_FOR_TEST, vocabulary_size)

    # Then
    check_tokenizer_mock.assert_called()
    fit_tokenizer_mock.assert_called_with(fake_path_to_texts, "a", vocabulary_size)


@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._check_is_fitted"
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._fit_tokenizer"
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._read_text_file",
    return_value=2,
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._tokenize_texts",
    return_value=1,
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._preprocess",
    return_value=(1, 2),
)
def test_language_modeling_dataset_transform_method_should_transformation_method_when_return_target_is_true(
    preprocess_mock,
    tokenize_texts_mock,
    read_text_file_mock,
    fit_tokenizer_mock,
    check_is_fitted_mock,
):
    # Given
    vocabulary_size = 10
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    language_modeling_dataset.set_tokenizer("a")
    language_modeling_dataset.fit(FAKE_PATH_FOR_TEST, vocabulary_size)

    # When
    _ = language_modeling_dataset.transform(FAKE_PATH_FOR_TEST, True)

    # Then
    check_is_fitted_mock.assert_called_with()
    read_text_file_mock.assert_called_with(FAKE_PATH_FOR_TEST)
    tokenize_texts_mock.assert_called_with(2)
    preprocess_mock.assert_called_with(1, 1, 1, True)


@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._check_is_fitted"
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._fit_tokenizer"
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._read_text_file",
    return_value=2,
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._tokenize_texts",
    return_value=1,
)
@patch(
    "language_modeling.domain.modeling.utils.data.dataset.LanguageModelingDataset._preprocess",
    return_value=(1, 2),
)
def test_language_modeling_dataset_transform_method_should_transformation_method_when_return_target_is_false(
    preprocess_mock,
    tokenize_texts_mock,
    read_text_file_mock,
    fit_tokenizer_mock,
    check_is_fitted_mock,
):
    # Given
    vocabulary_size = 10
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    language_modeling_dataset.set_tokenizer("a")
    language_modeling_dataset.fit(FAKE_PATH_FOR_TEST, vocabulary_size)

    # When
    _ = language_modeling_dataset.transform(FAKE_PATH_FOR_TEST, False)

    # Then
    check_is_fitted_mock.assert_called_with()
    read_text_file_mock.assert_called_with(FAKE_PATH_FOR_TEST)
    tokenize_texts_mock.assert_called_with(2)
    preprocess_mock.assert_called_with(1, 1, 1, False)


def test_language_model_dataset_fit_tokenizer_should_call_the_train_method_of_bpe_tokenizer():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train = MagicMock()
    language_modeling_dataset.set_tokenizer(tokenizer)

    # When
    language_modeling_dataset._fit_tokenizer(FAKE_PATH_FOR_TEST, tokenizer, 300)

    # Then
    tokenizer.train.assert_called_with(
        FAKE_PATH_FOR_TEST,
        300,
        special_tokens=[EOD_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN],
    )


def test_language_model_dataset_get_length_of_tokenized_texts_should_return_correct_sequence_of_ids_length():
    # Given
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    sequence_of_ids = [1, 2, 3]
    expected = 3

    # When
    output = language_modeling_dataset._get_length_of_tokenized_texts(sequence_of_ids)

    # Then
    assert output == expected


def test_language_model_dataset_get_total_number_of_batches_should_return_correct_sequence_of_ids_length():
    # Given
    batch_size = 5
    bptt = 2
    sequence_length = 100
    language_modeling_dataset = LanguageModelingDataset(batch_size, bptt)

    expected = 10

    # When
    output = language_modeling_dataset._get_total_number_of_batches(
        sequence_length, 5, 2
    )

    # Then
    assert output == expected


def test_language_model_dataset_truncate_sequence_of_ids_for_batch_processing_should_return_correct_output():
    # Given
    batch_size = 5
    bptt = 2
    total_number_of_batches = 10
    sequence_of_ids = [integer for integer in range(105)]
    language_modeling_dataset = LanguageModelingDataset(batch_size, bptt)

    # When
    output = language_modeling_dataset._truncate_sequence_of_ids_for_batch_processing(
        sequence_of_ids, batch_size, bptt, total_number_of_batches
    )

    # Then
    assert output == sequence_of_ids[0:100]


@patch("language_modeling.domain.modeling.utils.data.dataset.np.reshape")
def test_language_model_dataset_reshape_sequence_of_ids_for_batch_processing_should_call_reshape_function(
    reshape_mock,
):
    # Given
    batch_size = 5
    sequence_of_ids = [integer for integer in range(105)]
    language_modeling_dataset = LanguageModelingDataset(batch_size, 1)

    # When
    _ = language_modeling_dataset._reshape_sequence_of_ids_for_batch_processing(
        sequence_of_ids, batch_size
    )

    # Then
    reshape_mock.assert_called_with(sequence_of_ids, (batch_size, -1))


@patch("language_modeling.domain.modeling.utils.data.dataset.np.zeros_like")
def test_language_model_dataset_generate_empty_target_array_should_call_zeros_like_function(
    zeros_like_mock,
):
    # Given
    batch_size = 5
    sequence_of_ids = MagicMock()
    language_modeling_dataset = LanguageModelingDataset(batch_size, 1)

    # When
    _ = language_modeling_dataset._generate_empty_target_array(sequence_of_ids)

    # Then
    zeros_like_mock.assert_called_with(sequence_of_ids)
