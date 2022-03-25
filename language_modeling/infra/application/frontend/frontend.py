from typing import List, Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

FRONTEND_TITLE = "Financial news generator ⚡"
TEXT_INPUT_LABEL = "Seed str"
TEXT_INPUT_PLACEHOLDER = "Write something"
TEXT_INPUT_MAX_CHARS = 64

FIRST_COLUMN_SIZE = 0.7
SECOND_COLUMN_SIZE = 1

NUMBER_INPUT_LABEL = "Maximum number of tokens to generate"
NUMBER_INPUT_MINIMUM_VALUE = 1
NUMBER_INPUT_MAXIMUM_VALUE = 256

APPLICATION_DESCRIPTION_HEADER = "Application description"
APPLICATION_DESCRIPTION_CONTENT = "The aim of this application is to generate fake news."


class FrontEndBuilder:
    @staticmethod
    def _add_application_title() -> None:
        st.title(FRONTEND_TITLE)

    def _add_application_description(self) -> None:
        with st.container():
            self._add_application_header_description()
            self._add_application_description_content()
            columns = self._get_descriptions_columns()
            self._add_metrics(columns)

    @staticmethod
    def _add_application_header_description() -> None:
        st.header(APPLICATION_DESCRIPTION_HEADER)

    @staticmethod
    def _add_application_description_content() -> None:
        st.write(APPLICATION_DESCRIPTION_CONTENT)

    @staticmethod
    def _get_descriptions_columns() -> List[DeltaGenerator]:
        return st.columns(3)

    def _add_metrics(self, columns: List[DeltaGenerator]) -> None:
        self._add_metric(columns[0], "Tokens in training dataset", "XXXX")
        self._add_metric(columns[1], "Number of model parameters", "XXXX")
        self._add_metric(columns[2], "Model PPL on test", "XXXX")

    @staticmethod
    def _add_metric(column: DeltaGenerator, metric_label, metric_value) -> None:
        column.metric(metric_label, metric_value)

    @staticmethod
    def _add_application_columns() -> Tuple:
        return st.columns([FIRST_COLUMN_SIZE, SECOND_COLUMN_SIZE])

    @staticmethod
    def _add_text_input() -> str:
        return st.text_input(
            label=TEXT_INPUT_LABEL,
            placeholder=TEXT_INPUT_PLACEHOLDER,
            max_chars=TEXT_INPUT_MAX_CHARS
        )

    @staticmethod
    def _add_number_of_tokens_input() -> int:
        return st.number_input(
            label=NUMBER_INPUT_LABEL,
            min_value=NUMBER_INPUT_MINIMUM_VALUE,
            max_value=NUMBER_INPUT_MAXIMUM_VALUE
        )

    @staticmethod
    def _add_request_button() -> bool:
        return st.button(
            label="Generate sequence",
        )

    def build_frontend(self) -> None:
        self._add_application_title()
        self._add_application_description()
        container_one, container_two = self._add_application_columns()
        with container_one:
            maximum_number_of_tokens_to_generate = self._add_number_of_tokens_input()
            button_status = self._add_request_button()

        with container_two:
            text_input = self._add_text_input()

        if button_status:
            with st.spinner("Inference Model..."):
                import time
                time.sleep(2)
            st.header("Generated sequence")
            st.write(text_input)


if __name__ == "__main__":
    frontend_builder = FrontEndBuilder()
    frontend_builder.build_frontend()
