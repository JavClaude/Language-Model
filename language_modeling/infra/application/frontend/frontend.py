import os
from typing import List, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import requests

from language_modeling.infra.application.frontend import (
    FRONTEND_TITLE,
    TEXT_INPUT_LABEL,
    TEXT_INPUT_PLACEHOLDER,
    TEXT_INPUT_MAX_CHARS,
    FIRST_COLUMN_SIZE,
    SECOND_COLUMN_SIZE,
    NUMBER_INPUT_LABEL,
    NUMBER_INPUT_MINIMUM_VALUE,
    NUMBER_INPUT_MAXIMUM_VALUE,
    APPLICATION_DESCRIPTION_HEADER,
    APPLICATION_DESCRIPTION_CONTENT,
    NUMBER_OF_COLUMNS_TO_GENERATE,
    FIRST_COLUMN_METRIC_INDEX,
    SECOND_COLUMN_METRIC_INDEX,
    THIRD_COLUMN_METRIC_INDEX,
    GENERATE_SEQUENCE_BUTTON_LABEL,
    DEFAULT_METRIC_VALUE,
    FIRST_METRIC_LABEL_NAME,
    SECOND_METRIC_LABEL_NAME,
    THIRD_METRIC_LABEL_NAME,
    ERROR_MESSAGE_WHEN_TEXT_INPUT_IS_EMPTY,
    GENERATED_SEQUENCE_LABEL_NAME,
    RUNNING_INFERENCE_MESSAGE,
)
from language_modeling.infra.application.deep_learning_service.service import GENERATE_TEXT_ROOT_PATH

DEEP_LEARNING_SERVICE_PORT = os.environ["DEEP_LEARNING_SERVICE_PORT"]


class FrontEndBuilder:
    def build_frontend(self) -> None:
        self._add_application_title()
        self._add_application_description()
        (
            container_one,
            container_two,
        ) = self._get_application_containers_for_input_field_and_buttons()
        (
            maximum_number_of_tokens_to_generate,
            generate_sequence_status_button,
        ) = self._add_application_first_container(container_one)
        text_seed = self._add_application_second_container(container_two)

        if generate_sequence_status_button:
            if text_seed:
                generated_text = (
                    self._send_request_to_the_backend_deep_learning_service(
                        maximum_number_of_tokens_to_generate, text_seed
                    )
                )
                self._add_application_header_for_generated_text()
                self._add_application_generated_text_field(generated_text)
            else:
                self._add_application_error_message_when_text_seed_is_empty()

    @staticmethod
    def _add_application_title() -> None:
        st.title(FRONTEND_TITLE)

    def _add_application_description(self) -> None:
        with st.container():
            self._add_application_description_header()
            self._add_application_description_content()
            columns = self._get_application_description_columns()
            self._add_application_description_metrics(columns)

    @staticmethod
    def _add_application_description_header() -> None:
        st.header(APPLICATION_DESCRIPTION_HEADER)

    @staticmethod
    def _add_application_description_content() -> None:
        st.write(APPLICATION_DESCRIPTION_CONTENT)

    @staticmethod
    def _get_application_description_columns() -> List[DeltaGenerator]:
        return st.columns(NUMBER_OF_COLUMNS_TO_GENERATE)

    def _add_application_description_metrics(
            self, columns: List[DeltaGenerator]
    ) -> None:
        self._add_metric(
            columns[FIRST_COLUMN_METRIC_INDEX],
            FIRST_METRIC_LABEL_NAME,
            DEFAULT_METRIC_VALUE,
        )
        self._add_metric(
            columns[SECOND_COLUMN_METRIC_INDEX],
            SECOND_METRIC_LABEL_NAME,
            DEFAULT_METRIC_VALUE,
        )
        self._add_metric(
            columns[THIRD_COLUMN_METRIC_INDEX],
            THIRD_METRIC_LABEL_NAME,
            DEFAULT_METRIC_VALUE,
        )

    @staticmethod
    def _add_metric(
            column: DeltaGenerator, metric_label: str, metric_value: str
    ) -> None:
        column.metric(metric_label, metric_value)

    @staticmethod
    def _get_application_containers_for_input_field_and_buttons() -> List[
        DeltaGenerator
    ]:
        return st.columns([FIRST_COLUMN_SIZE, SECOND_COLUMN_SIZE])

    def _add_application_first_container(
            self, container: DeltaGenerator
    ) -> Tuple[int, bool]:
        with container:
            maximum_number_of_tokens_to_generate = (
                self._add_application_number_of_tokens_input_field()
            )
            generate_sequence_status_button = (
                self._add_application_generate_sequence_button()
            )
        return maximum_number_of_tokens_to_generate, generate_sequence_status_button

    @staticmethod
    def _add_application_number_of_tokens_input_field() -> int:
        return st.number_input(
            label=NUMBER_INPUT_LABEL,
            min_value=NUMBER_INPUT_MINIMUM_VALUE,
            max_value=NUMBER_INPUT_MAXIMUM_VALUE,
        )

    @staticmethod
    def _add_application_generate_sequence_button() -> bool:
        return st.button(label=GENERATE_SEQUENCE_BUTTON_LABEL)

    def _add_application_second_container(self, container: DeltaGenerator) -> str:
        with container:
            text_seed = self._add_application_text_input_field()
        return text_seed

    @staticmethod
    def _add_application_text_input_field() -> str:
        return st.text_input(
            label=TEXT_INPUT_LABEL,
            placeholder=TEXT_INPUT_PLACEHOLDER,
            max_chars=TEXT_INPUT_MAX_CHARS,
        )

    @staticmethod
    def _add_application_error_message_when_text_seed_is_empty() -> None:
        st.error(ERROR_MESSAGE_WHEN_TEXT_INPUT_IS_EMPTY)

    @staticmethod
    def _send_request_to_the_backend_deep_learning_service(
            maximum_number_of_tokens_to_generate: int, text_seed: str
    ) -> str:
        with st.spinner(RUNNING_INFERENCE_MESSAGE):
            # perform request here
            response = requests.post(
                "{}:{}{}".format(
                    "deep-learning",
                    DEEP_LEARNING_SERVICE_PORT,
                    GENERATE_TEXT_ROOT_PATH,
                ),
                json={
                    "seed_str": text_seed,
                    "maximum_sequence_length": maximum_number_of_tokens_to_generate,
                    "top_k_word": 2
                })
        return response.content

    @staticmethod
    def _add_application_header_for_generated_text() -> None:
        st.header(GENERATED_SEQUENCE_LABEL_NAME)

    @staticmethod
    def _add_application_generated_text_field(generated_text: str) -> None:
        st.write(generated_text)


if __name__ == "__main__":
    frontend_builder = FrontEndBuilder()
    frontend_builder.build_frontend()
