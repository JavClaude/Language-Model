import re

from scrapy import Spider
from scrapy.exceptions import DropItem
from w3lib.html import replace_escape_chars, replace_tags, strip_html5_whitespace

from language_modeling.domain.scraper.pipelines import (
    ENCODING_VALUE,
    SPACES_PATTERN_REGEX_TO_CATCH,
    EMPTY_STRING,
)
from language_modeling.domain.scraper.spiders.boursorama.data_model import BaseItem


class CleanHtmlPipeline:
    def process_item(self, item: BaseItem, spider: Spider) -> BaseItem:
        if self._check_empty_items(item):
            raise DropItem()
        return self._process_item_fields(item)

    def _process_item_fields(self, item: BaseItem) -> BaseItem:
        item_fields = item.__dict__
        for key, value in item_fields.items():
            if value is not None:
                item_fields[key] = self._apply_item_processing(value)
        return item.__class__(**item_fields)

    @staticmethod
    def _check_empty_items(item: BaseItem) -> bool:
        return all(list(map(lambda x: x == None, item.__dict__.values())))

    def _apply_item_processing(self, raw_html_content: str) -> str:
        html_content = self._replace_html_tags(raw_html_content)
        html_content = self._replace_escape_chars(html_content)
        html_content = self._strip_html5_whitespace(html_content)
        html_content = self._replace_html_tags(html_content)
        return html_content

    @staticmethod
    def _replace_html_tags(raw_html_content: str) -> str:
        return replace_tags(raw_html_content, encoding=ENCODING_VALUE)

    @staticmethod
    def _replace_escape_chars(raw_html_content: str) -> str:
        return replace_escape_chars(raw_html_content, encoding=ENCODING_VALUE)

    @staticmethod
    def _strip_html5_whitespace(raw_html_content: str) -> str:
        return strip_html5_whitespace(raw_html_content)

    @staticmethod
    def _replace_multiple_spaces(raw_html_content: str) -> str:
        return re.sub(SPACES_PATTERN_REGEX_TO_CATCH, EMPTY_STRING, raw_html_content)
