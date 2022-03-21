from scrapy import Spider
from scrapy.http import Response

from language_modeling.domain.scraper.spiders.boursorama.base import (
    BaseBoursoramaSpider,
)
from language_modeling.domain.scraper.spiders.boursorama.economic_news import (
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_SOURCE_NAME_XPATH,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_DATE_XPATH,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_TITLE,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_CONTENT,
)
from language_modeling.domain.scraper.spiders.boursorama.data_model import (
    BoursoramaNews,
)


class BoursoramaEconomicNewsSpider(BaseBoursoramaSpider, Spider):
    name = BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME
    start_urls = BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS
    last_page_url_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH
    all_news_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH

    def _parse_news_content(self, response: Response) -> BoursoramaNews:
        yield BoursoramaNews(
            source_name=self._extract_data_from_response(
                response,
                BOURSORAMA_ECONOMIC_NEWS_SPIDER_SOURCE_NAME_XPATH,
            ),
            date=self._extract_data_from_response(
                response,
                BOURSORAMA_ECONOMIC_NEWS_SPIDER_DATE_XPATH,
            ),
            news_title=self._extract_data_from_response(
                response,
                BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_TITLE,
            ),
            news_text=self._extract_data_from_response(
                response,
                BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_CONTENT,
            ),
        )
