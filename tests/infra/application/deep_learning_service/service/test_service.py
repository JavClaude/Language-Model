from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from language_modeling.infra.application.deep_learning_service.service.service import DeepLearningService, GENERATE_TEXT_ROOT_PATH


@patch("language_modeling.infra.application.deep_learning_service.service.service.FastAPI", return_value="API")
@patch("language_modeling.infra.application.deep_learning_service.service.service.GreedyDecoder")
def test_deep_learning_service_get_api_should_correct_output(
        greedy_decoder_mock,
        fastapi_mock
):
    # Given
    tokenizer = "tokenizer"
    model = "model"
    expected = "API"
    service = DeepLearningService(tokenizer, model)

    # When
    output = service.get_api()

    # Then
    greedy_decoder_mock.assert_called_with(tokenizer, model)
    fastapi_mock.assert_called()
    assert output == expected


@patch("language_modeling.infra.application.deep_learning_service.service.service.FastAPI")
@patch("language_modeling.infra.application.deep_learning_service.service.service.GreedyDecoder")
@patch("language_modeling.infra.application.deep_learning_service.service.service.DeepLearningService._add_generate_text_endpoint")
def test_deep_learning_service_build_api_should_call_add_generate_text_endpoint_method(
        add_generate_text_endpoint_mock,
        greedy_decoder_mock,
        fastapi_mock
):
    # Given
    tokenizer = "tokenizer"
    model = "model"
    service = DeepLearningService(tokenizer, model)

    # When
    service.build_api()

    # Then
    fastapi_mock.assert_called()
    greedy_decoder_mock.assert_called()
    add_generate_text_endpoint_mock.assert_called()


# @patch("language_modeling.infra.application.deep_learning_service.service.service.GreedyDecoder")
# def test_deep_learning_service_generate_text_root_return_is_correctly_called(
#         greedy_decoder_mock
# ):
#     # Given
#     tokenizer = "tokenizer"
#     model = "model"
#     greedy_decoder_mock.generate_text = MagicMock(return_value="Fake text for test")
#
#     service = DeepLearningService(tokenizer, model)
#     service.build_api()
#     test_client = TestClient(service.get_api())
#
#     # When
#     output = test_client.post(
#         GENERATE_TEXT_ROOT_PATH,
#         json={
#             "seed_str": "Les taux",
#             "maximum_sequence_length": 10,
#             "top_k_word": 3
#         }
#     )
#
#     # Then
#     greedy_decoder_mock.generate_text.assert_called()

