from unittest.mock import patch, mock_open

from language_modeling.infra.application.deep_learning_service.utils.load_artifacts import ArtifactsLoader


@patch("language_modeling.infra.application.deep_learning_service.utils.load_artifacts.pickle.load")
def test_artifacts_loader_load_preprocessor_and_model_method_should_call_pickle_load_function_with_correct_arguments(
        load_mock
):
    # Given
    fake_path_to_artifacts = "fake/path/to/model"
    fake_content = "fake_content"
    artifacts_loader = ArtifactsLoader()

    # When
    with patch("builtins.open", new_callable=mock_open()) as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_content
        artifacts_loader.load_preprocessor_and_model(fake_path_to_artifacts)

    # Then
    load_mock.assert_called_with(fake_content)
