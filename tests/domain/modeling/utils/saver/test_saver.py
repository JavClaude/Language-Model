from unittest.mock import patch, mock_open

from language_modeling.domain.modeling.utils.saver.saver import Saver


@patch("language_modeling.domain.modeling.utils.saver.saver.pickle.dump")
@patch("language_modeling.domain.modeling.utils.saver.saver.Saver._get_joined_path")
def test_saver_save_preprocessor_and_model_should_call_the_pickle_dump_function_with_correct_arguments(
    get_joined_path, dump_mock
):
    # Given
    test_log_dir = "test"
    artifact_name = "preprocessor_and_model.bin"
    saver = Saver(test_log_dir)
    dataset = "dataset"
    model = "model"
    tuple_of_dataset_and_model = tuple((dataset, model))

    # When
    with patch("builtins.open", new_callable=mock_open()) as mock_writer:
        mock_writer.return_value.__enter__.return_value = "b"
        saver.save_preprocessor_and_model(dataset, model)

    # Then
    get_joined_path.assert_called_with(test_log_dir, artifact_name)
    dump_mock.assert_called_with(tuple_of_dataset_and_model, "b")
