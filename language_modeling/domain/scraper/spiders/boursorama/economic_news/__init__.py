BOURSORAMA_ECONOMIC_NEWS_SPIDER_NAME = "boursorama_economic_news_spider"
BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URL = (
    "https://www.boursorama.com/actualite-economique/"
)
BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URLS = [BOURSORAMA_ECONOMIC_NEWS_SPIDER_START_URL]
BOURSORAMA_ECONOMIC_NEWS_SPIDER_LAST_URL_XPATH = "//*[@id='main-content']/div/div[1]/div[4]/div[1]/div[3]//div[@class='c-pagination']/a/@href"
BOURSORAMA_ECONOMIC_NEWS_SPIDER_ALL_NEWS_XPATH = "//*[@id='main-content']/div/div[1]/div[4]/div[1]/div[3]/div[1]/div[2]/ul/li/div[@class='c-list-details-news__title']/a/@href"

BOURSORAMA_ECONOMIC_NEWS_SPIDER_SOURCE_NAME_XPATH = "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/strong/text()"
BOURSORAMA_ECONOMIC_NEWS_SPIDER_DATE_XPATH = "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/div/span[3]/text()"
BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_TITLE = "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/h1/text()"
BOURSORAMA_ECONOMIC_NEWS_SPIDER_NEWS_CONTENT = "//*[@id='main-content']/div/div/div[1]/div[1]/div[2]/div[1]/div[2]/div[5]"