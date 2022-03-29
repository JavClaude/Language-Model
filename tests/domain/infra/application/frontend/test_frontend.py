from unittest.mock import patch, call, MagicMock

from language_modeling.infra.application.frontend.frontend import FrontEndBuilder

from language_modeling.infra.application.frontend import (
    FRONTEND_TITLE,
    TEXT_INPUT_LABEL,
    TEXT_INPUT_PLACEHOLDER,
    TEXT_INPUT_MAX_CHARS,
    NUMBER_INPUT_LABEL,
    NUMBER_INPUT_MINIMUM_VALUE,
    NUMBER_INPUT_MAXIMUM_VALUE,
    APPLICATION_DESCRIPTION_HEADER,
    APPLICATION_DESCRIPTION_CONTENT,
    NUMBER_OF_COLUMNS_TO_GENERATE,
    GENERATE_SEQUENCE_BUTTON_LABEL,
    DEFAULT_METRIC_VALUE,
    FIRST_METRIC_LABEL_NAME,
    SECOND_METRIC_LABEL_NAME,
    THIRD_METRIC_LABEL_NAME,
    ERROR_MESSAGE_WHEN_TEXT_INPUT_IS_EMPTY,
    GENERATED_SEQUENCE_LABEL_NAME,
    RUNNING_INFERENCE_MESSAGE,
)


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_title"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._get_application_containers_for_input_field_and_buttons",
    return_value=("c1", "c2"),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_first_container",
    return_value=(5, True),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_second_container",
    return_value="text seed for test",
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._send_request_to_the_backend_deep_learning_service",
    return_value="generated text for test",
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_header_for_generated_text"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_generated_text_field"
)
def test_frontend_builder_build_frontend_should_call_other_building_methods_when_button_status_is_true_and_text_seed_not_empty(
    add_application_generated_text_field_mock,
    add_application_header_for_generated_text_mock,
    send_request_to_the_backend_deep_learning_service_mock,
    add_application_second_container_mock,
    add_application_first_container_mock,
    get_application_containers_for_input_field_and_buttons_mock,
    add_application_description_mock,
    add_application_title_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder.build_frontend()

    # Then
    add_application_generated_text_field_mock.assert_called()
    add_application_header_for_generated_text_mock.assert_called()
    send_request_to_the_backend_deep_learning_service_mock.assert_called_with(
        5, "text seed for test"
    )
    add_application_second_container_mock.assert_called_with("c2")
    add_application_first_container_mock.assert_called_with("c1")
    get_application_containers_for_input_field_and_buttons_mock.assert_called()
    add_application_description_mock.assert_called()
    add_application_title_mock.assert_called()


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_title"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._get_application_containers_for_input_field_and_buttons",
    return_value=("c1", "c2"),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_first_container",
    return_value=(5, True),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_second_container",
    return_value="",
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_error_message_when_text_seed_is_empty"
)
def test_frontend_builder_build_frontend_should_call_other_building_methods_when_button_status_is_true_and_text_seed_empty(
    add_application_error_message_when_text_seed_is_empty_mock,
    add_application_second_container_mock,
    add_application_first_container_mock,
    get_application_containers_for_input_field_and_buttons_mock,
    add_application_description_mock,
    add_application_title_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder.build_frontend()

    # Then
    add_application_second_container_mock.assert_called_with("c2")
    add_application_first_container_mock.assert_called_with("c1")
    get_application_containers_for_input_field_and_buttons_mock.assert_called()
    add_application_description_mock.assert_called()
    add_application_title_mock.assert_called()
    add_application_error_message_when_text_seed_is_empty_mock()


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_title"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._get_application_containers_for_input_field_and_buttons",
    return_value=("c1", "c2"),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_first_container",
    return_value=(5, False),
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_second_container",
    return_value="",
)
def test_frontend_builder_build_frontend_should_call_other_building_methods_when_button_status_is_false_and_text_seed_empty(
    add_application_second_container_mock,
    add_application_first_container_mock,
    get_application_containers_for_input_field_and_buttons_mock,
    add_application_description_mock,
    add_application_title_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder.build_frontend()

    # Then
    add_application_second_container_mock.assert_called_with("c2")
    add_application_first_container_mock.assert_called_with("c1")
    get_application_containers_for_input_field_and_buttons_mock.assert_called()
    add_application_description_mock.assert_called()
    add_application_title_mock.assert_called()


@patch("language_modeling.infra.application.frontend.frontend.st.title")
def test_frontend_builder_add_application_title_should_call_st_title_with_correct_title(
    title_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_title()

    # Then
    title_mock.assert_called_with(FRONTEND_TITLE)


@patch("language_modeling.infra.application.frontend.frontend.st.container")
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description_header"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description_content"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._get_application_description_columns",
    return_value="a",
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_description_metrics"
)
def test_frontend_builder_add_application_description_should_call_other_methods(
    add_application_description_metrics_mock,
    get_application_description_columns_mock,
    add_application_description_content_mock,
    add_application_description_header_mock,
    container_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_description()

    # Then
    container_mock.assert_called()
    add_application_description_header_mock.assert_called()
    add_application_description_content_mock.assert_called()
    get_application_description_columns_mock.assert_called()
    add_application_description_metrics_mock.assert_called_with("a")


@patch("language_modeling.infra.application.frontend.frontend.st.header")
def test_frontend_builder_add_application_description_header_should_call_st_header_with_correct_header(
    header_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_description_header()

    # Then
    header_mock.assert_called_with(APPLICATION_DESCRIPTION_HEADER)


@patch("language_modeling.infra.application.frontend.frontend.st.write")
def test_frontend_builder_add_application_description_content_should_call_st_write_with_correct_description_content(
    write_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_description_content()

    # Then
    write_mock.assert_called_with(APPLICATION_DESCRIPTION_CONTENT)


@patch("language_modeling.infra.application.frontend.frontend.st.columns")
def test_frontend_builder_get_application_description_columns_should_call_st_columns_with_correct_nb_of_col_to_generate(
    columns_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._get_application_description_columns()

    # Then
    columns_mock.assert_called_with(NUMBER_OF_COLUMNS_TO_GENERATE)


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_metric"
)
def test_frontend_builder_add_application_description_metrics_should_call_add_metric_with_correct_parameters(
    add_metric_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()
    first_column = "a"
    second_column = "b"
    third_column = "c"
    fake_columns = [first_column, second_column, third_column]

    # When
    frontend_builder._add_application_description_metrics(fake_columns)

    # Then
    add_metric_mock.assert_has_calls(
        [
            call(first_column, FIRST_METRIC_LABEL_NAME, DEFAULT_METRIC_VALUE),
            call(second_column, SECOND_METRIC_LABEL_NAME, DEFAULT_METRIC_VALUE),
            call(third_column, THIRD_METRIC_LABEL_NAME, DEFAULT_METRIC_VALUE),
        ]
    )


def test_frontend_builder_add_metric_should_call_the_column_metric_method_with_correct_parameters():
    # Given
    frontend_builder = FrontEndBuilder()
    column_mock = MagicMock()
    metric_label = "a"
    metric_value = "b"

    # When
    frontend_builder._add_metric(column_mock, metric_label, metric_value)

    # Then
    column_mock.metric.assert_called_with(metric_label, metric_value)


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_number_of_tokens_input_field"
)
@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_generate_sequence_button"
)
def test_frontend_builder_add_application_first_container_should_call_other_methods_with_correct_parameters(
    add_application_generate_sequence_button_mock,
    add_application_number_of_tokens_input_field_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()
    container_mock = MagicMock()

    # When
    frontend_builder._add_application_first_container(container_mock)

    # Then
    container_mock.__enter__.assert_called()
    add_application_number_of_tokens_input_field_mock.assert_called()
    add_application_generate_sequence_button_mock.assert_called()


@patch("language_modeling.infra.application.frontend.frontend.st.number_input")
def test_frontend_builder_add_application_number_of_tokens_input_field_should_call_st_number_input_with_correct_params(
    number_input_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_number_of_tokens_input_field()

    # Then
    number_input_mock.assert_called_with(
        label=NUMBER_INPUT_LABEL,
        min_value=NUMBER_INPUT_MINIMUM_VALUE,
        max_value=NUMBER_INPUT_MAXIMUM_VALUE,
    )


@patch("language_modeling.infra.application.frontend.frontend.st.button")
def test_frontend_builder_add_application_generate_sequence_button_should_call_st_button_input_with_correct_label(
    button_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_generate_sequence_button()

    # Then
    button_mock.assert_called_with(label=GENERATE_SEQUENCE_BUTTON_LABEL)


@patch(
    "language_modeling.infra.application.frontend.frontend.FrontEndBuilder._add_application_text_input_field"
)
def test_frontend_builder_add_application_second_container_should_call__add_application_text_input_field(
    add_application_second_container_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()
    container_mock = MagicMock()

    # When
    frontend_builder._add_application_second_container(container_mock)

    # Then
    container_mock.__enter__.assert_called()
    add_application_second_container_mock.assert_called()


@patch("language_modeling.infra.application.frontend.frontend.st.text_input")
def test_frontend_builder_add_application_text_input_field_should_call__add_application_text_input_field(
    add_application_text_input_field_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_text_input_field()

    # Then
    add_application_text_input_field_mock.assert_called_with(
        label=TEXT_INPUT_LABEL,
        placeholder=TEXT_INPUT_PLACEHOLDER,
        max_chars=TEXT_INPUT_MAX_CHARS,
    )


@patch("language_modeling.infra.application.frontend.frontend.st.error")
def test_frontend_builder_add_application_error_message_when_text_seed_is_empty_mock_should_call_st_error_with_correct_error_message(
    error_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_error_message_when_text_seed_is_empty()

    # Then
    error_mock.assert_called_with(ERROR_MESSAGE_WHEN_TEXT_INPUT_IS_EMPTY)


@patch("language_modeling.infra.application.frontend.frontend.st.spinner")
@patch("language_modeling.infra.application.frontend.frontend.time.sleep")
def test_frontend_builder_send_request_to_the_backend_deep_learning_service_should_st_spinner_and_time_sleep(
    sleep_mock, spinner_mock
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._send_request_to_the_backend_deep_learning_service(1, "")

    # Then
    spinner_mock.assert_called_with(RUNNING_INFERENCE_MESSAGE)
    sleep_mock.assert_called_with(3)


@patch("language_modeling.infra.application.frontend.frontend.st.header")
def test_frontend_builder_add_application_header_for_generated_text_should_st_spinner_and_time_sleep(
    header_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()

    # When
    frontend_builder._add_application_header_for_generated_text()

    # Then
    header_mock.assert_called_with(GENERATED_SEQUENCE_LABEL_NAME)


@patch("language_modeling.infra.application.frontend.frontend.st.write")
def test_frontend_builder_add_application_generated_text_field_should_st_spinner_and_time_sleep(
    write_mock,
):
    # Given
    frontend_builder = FrontEndBuilder()
    generated_text = ""
    # When
    frontend_builder._add_application_generated_text_field(generated_text)

    # Then
    write_mock.assert_called_with(generated_text)
