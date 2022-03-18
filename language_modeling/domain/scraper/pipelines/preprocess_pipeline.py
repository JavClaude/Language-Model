import re

from w3lib.html import replace_escape_chars, replace_tags, strip_html5_whitespace

from language_modeling.domain.scraper.pipelines import (
    ENCODING_VALUE,
    SPACES_PATTERN_REGEX_TO_CATCH,
    EMPTY_STRING,
)


class CleanHtmlPipeline:
    def _replace_html_tags(self, raw_html_content: str) -> str:
        return replace_tags(raw_html_content, encoding=ENCODING_VALUE)

    def _replace_escape_chars(self, raw_html_content: str) -> str:
        return replace_escape_chars(raw_html_content, encoding=ENCODING_VALUE)

    def _strip_html5_whitespace(self, raw_html_content: str) -> str:
        return strip_html5_whitespace(raw_html_content)

    def _replace_multiple_spaces(self, raw_html_content: str) -> str:
        return re.sub(SPACES_PATTERN_REGEX_TO_CATCH, EMPTY_STRING, raw_html_content)
