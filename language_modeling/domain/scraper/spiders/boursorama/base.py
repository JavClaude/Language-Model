from abc import abstractmethod
from typing import Tuple, List, Any

from scrapy import Request
from scrapy.http import Response


class BaseBoursoramaSpider:
    def _get_page_str_and_maximum_number_of_pages(
        self, response: Response, last_page_url_xpath: str
    ) -> Tuple[str, int]:
        last_url = self._get_last_page_url(response, last_page_url_xpath)
        (
            page_str,
            maximum_number_of_pages,
        ) = self._get_page_str_and_maximum_number_of_pages_from_the_last_page_url(
            last_url
        )
        return page_str, maximum_number_of_pages

    @staticmethod
    def _get_last_page_url(response: Response, last_page_url_xpath: str) -> str:
        return response.xpath(last_page_url_xpath).getall()[-1]

    @staticmethod
    def _get_page_str_and_maximum_number_of_pages_from_the_last_page_url(
        last_page_url: str,
    ) -> Tuple[str, int]:
        page_str, maximum_number_of_pages = last_page_url.split("/")[-1].split("-")
        return page_str, int(maximum_number_of_pages)

    @staticmethod
    def _get_next_page_url_to_parse(
        response: Response, page_str: str, page_number: int
    ) -> str:
        join_page_str_and_page_number = page_str + "-" + str(page_number)
        return response.urljoin(join_page_str_and_page_number)

    def parse(self, response: Response) -> Request:
        (
            page_str,
            maximum_number_of_pages,
        ) = self._get_page_str_and_maximum_number_of_pages(
            response, self.last_page_url_xpath
        )

        for page_number in range(1, maximum_number_of_pages + 1):
            yield Request(
                self._get_next_page_url_to_parse(response, page_str, page_number),
                callback=self._parse_all_news_page,
            )

    def _parse_all_news_page(self, response: Response) -> Request:
        all_news_url = self._get_all_news_url(response, self.all_news_xpath)
        for news_url in all_news_url:
            yield Request(
                self._get_next_news_url_to_parse(response, news_url),
                callback=self._parse_news_content,
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
