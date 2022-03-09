import scrapy
from scrapy.http import request
from ..items import BFMItem

class BFM_Economie_Spider(scrapy.Spider):
    name = "BFM_Economie_Spider"
    collection_name = "BFM_Eco"

    start_urls = [
        "https://www.bfmtv.com/archives/economie/"
    ]

    def parse(self, response):

        all_months = response.xpath("//*[@id='archives_block']/article/ul/li/a/@href").getall()

        for i, month in enumerate(all_months):
            suffix_url = all_months[i].split("economie/")[-1]
            month_content_url = response.urljoin(suffix_url)
            yield scrapy.Request(url=month_content_url, meta={"crawl_once": False}, callback=self.parse_month)

    def parse_month(self, response):

        page_content = response.xpath("//*[@id='main_wrapper']/div/div[1]/section/div/article/a/@href").getall()

        for i, url in enumerate(page_content):
            yield scrapy.Request(url=url, meta={"crawl_once": True}, callback=self.parse_content)

        next_page = response.xpath("//*[@id='main_wrapper']/div/div[1]/ul/li[@class='pagination-btn']/a[@rel='next']").get()

        if next_page is not None:
            next_page_url = next_page.split("economie/")[-1]
            yield scrapy.Request(url=self.start_urls[0] + next_page_url, meta={"crawl_once" : False}, callback=self.parse_month)

    def parse_content(self, response):
        category = "Economie"
        date = response.xpath("//*[@id='content_scroll_start']/time/text()").get()
        header = response.xpath("//*[@id='content_progress']/div/div/div[1]/text()").get()
        title = response.xpath("//*[@id='contain_title']/text()").get()
        text = response.xpath("//*[@id='content_progress']/div/div/*[not(@class='chapo')]").getall()
        text = " ".join(elt for elt in text)
        yield BFMItem(category, date, header, title, text)