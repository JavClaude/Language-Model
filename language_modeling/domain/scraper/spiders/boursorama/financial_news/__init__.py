BOURSORAMA_FINANCIAL_NEWS_SPIDER_NAME = "boursorama-financial-news-spider"
BOURSORAMA_FINANCIAL_NEWS_SPIDER_START_URL = (
    "https://www.boursorama.com/bourse/actualites/finances/"
)
BOURSORAMA_FINANCIAL_NEWS_SPIDER_START_URLS = [
    BOURSORAMA_FINANCIAL_NEWS_SPIDER_START_URL
]

BOURSORAMA_FINANCIAL_NEWS_SPIDER_LAST_URL_XPATH = "//*[@id='main-content']/div/div[1]/div[4]/div[1]/article/div[2]/div/div/div[2]/div[2]/div/a/@href"
BOURSORAMA_FINANCIAL_NEWS_SPIDER_ALL_NEWS_XPATH = "//*[@id='main-content']/div/div[1]/div[4]/div[1]/article/div[2]/div/div/div[2]/div[1]/div/ul/li/div/p/a/@href"

BOURSORAMA_FINANCIAL_NEWS_SPIDER_SOURCE_NAME_XPATH = "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/strong/text()"
BOURSORAMA_FINANCIAL_NEWS_SPIDER_DATE_XPATH = (
    "/html/body/main/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/span[3]/text()"
)
BOURSORAMA_FINANCIAL_NEWS_SPIDER_NEWS_TITLE_XPATH = (
    "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1"
)
BOURSORAMA_FINANCIAL_NEWS_SPIDER_NEWS_CONTENT_XPATH = (
    "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/div[5]"
)
