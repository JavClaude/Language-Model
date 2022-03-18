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
)
from language_modeling.domain.scraper.spiders.boursorama.data_model import (
    BoursoramaEconomicNews,
)


class BoursoramaEconomicNewsSpider(BaseBoursoramaSpider, Spider):
    name = BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME
    start_urls = BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS
    last_page_url_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH
    all_news_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH

    def _parse_news_content(self, response: Response) -> BoursoramaEconomicNews:
        yield BoursoramaEconomicNews(
            source_name=self._extract_data_from_response(
                response,
                "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/strong/text()",
            ),
            date=self._extract_data_from_response(
                response,
                "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/span[2]/text()",
            ),
            news_title=self._extract_data_from_response(
                response,
                "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/text()",
            ),
            news_header=None,
            news_text=self._extract_data_from_response(
                response,
                "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/div[5]",
            ),
        )
