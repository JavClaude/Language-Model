from unittest.mock import patch

import pytest

from language_model.domain.modeling.data import (
    EOD_TOKEN,
    SOS_TOKEN,
    TOKENIZER_FIELD,
    CHECK_TOKENIZER_ERROR_MESSAGE,
    IS_FITTED_MESSAGE
)
from language_model.domain.modeling.data.dataset import LanguageModelingDataset
from language_model.domain.modeling.data.errors import MissingTokenizerError, NotFittedDatasetError


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
        len(language_modeling_dataset)

    # Then
    assert isinstance(exc_info.value, NotFittedDatasetError)
    assert str(exc_info.value) == IS_FITTED_MESSAGE


@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._check_tokenizer")
@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._fit_tokenizer")
@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._read_text_file", return_value=2)
@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._tokenize_texts", return_value=1)
@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._orchestrate_dataset_creation", return_value=(1, 2))
@patch("language_model.domain.modeling.data.dataset.LanguageModelingDataset._get_sequence_of_ids_length")
def test_language_modeling_dataset_fit_method_should_call_other_fitting_method(
        get_sequence_of_ids_length_mock,
        orchestrate_dataset_creation_mock,
        tokenize_texts_mock,
        read_text_file_mock,
        fit_tokenizer_mock,
        check_tokenizer_mock
):
    # Given
    fake_path_to_texts = "fake/path/to/text"
    vocabulary_size = 10
    language_modeling_dataset = LanguageModelingDataset(1, 1)
    language_modeling_dataset.set_tokenizer("a")

    # When
    language_modeling_dataset.fit(fake_path_to_texts, vocabulary_size)

    # Then
    check_tokenizer_mock.assert_called()
    fit_tokenizer_mock.assert_called_with(fake_path_to_texts, "a", vocabulary_size)
    read_text_file_mock.assert_called_with(fake_path_to_texts)
    tokenize_texts_mock.assert_called_with(2)
    orchestrate_dataset_creation_mock.assert_called_with(1, 1, 1)
    get_sequence_of_ids_length_mock.assert_called_with(1)
