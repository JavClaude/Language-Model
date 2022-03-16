from abc import abstractmethod
from typing import Any, List, Tuple

from scrapy import Spider, Request
from scrapy.http import Response

from language_modeling.domain.scraper.spiders.boursorama import (
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH,
    BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH
)
from language_modeling.domain.scraper.spiders.boursorama.data_model import BoursoramaEconomicNews


class BaseBoursoramaSpider:
    def _get_page_str_and_maximum_number_of_pages(self, response: Response, last_page_url_xpath: str) -> Tuple[str, int]:
        last_url = self._get_last_page_url(response, last_page_url_xpath)
        page_str, maximum_number_of_pages = self._get_page_str_and_maximum_number_of_pages_from_the_last_page_url(last_url)
        return page_str, maximum_number_of_pages

    @staticmethod
    def _get_last_page_url(response: Response, last_page_url_xpath: str) -> str:
        return response.xpath(last_page_url_xpath).getall()[-1]

    @staticmethod
    def _get_page_str_and_maximum_number_of_pages_from_the_last_page_url(last_page_url: str) -> Tuple[str, int]:
        page_str, maximum_number_of_pages = last_page_url.split("/")[-1].split("-")
        return page_str, int(maximum_number_of_pages)

    @staticmethod
    def _get_next_page_url_to_parse(response: Response, page_str: str, page_number: int) -> str:
        join_page_str_and_page_number = page_str + "-" + str(page_number)
        return response.urljoin(join_page_str_and_page_number)

    def parse(self, response: Response) -> Request:
        page_str, maximum_number_of_pages = self._get_page_str_and_maximum_number_of_pages(response, self.last_page_url_xpath)

        for page_number in range(1, maximum_number_of_pages + 1):
            yield Request(
                self._get_next_page_url_to_parse(response, page_str, page_number),
                callback=self._parse_all_news_page
            )

    def _parse_all_news_page(self, response: Response) -> Request:
        all_news_url = self._get_all_news_url(response, self.all_news_xpath)
        for news_url in all_news_url:
            yield Request(
                self._get_next_news_url_to_parse(response, news_url),
                callback=self._parse_news_content
            )

    @staticmethod
    def _get_all_news_url(response: Response, all_news_xpath: str) -> List[str]:
        return response.xpath(all_news_xpath).getall()

    @staticmethod
    def _get_next_news_url_to_parse(response: Response, new_url: str) -> str:
        return response.urljoin(new_url)

    def _extract_data_from_response(self, response: Response, xpath: str) -> str:
        return response.xpath(xpath).get()

    @abstractmethod
    def _parse_news_content(self, response: Response) -> Any:
        pass


class BoursoramaEconomicNewsSpider(BaseBoursoramaSpider, Spider):
    name = BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME
    start_urls = BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS
    last_page_url_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH
    all_news_xpath = BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH

    def _parse_news_content(self, response: Response) -> BoursoramaEconomicNews:
        yield BoursoramaEconomicNews(
            source_name=self._extract_data_from_response(response, "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/strong/text()"),
            date=self._extract_data_from_response(response, "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/span[2]/text()"),
            news_title=self._extract_data_from_response(response, "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/text()"),
            news_header=None,
            news_text=self._extract_data_from_response(response, "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/div[5]")
        )
