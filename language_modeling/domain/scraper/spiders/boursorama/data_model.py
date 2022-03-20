from dataclasses import dataclass


@dataclass
class BaseItem:
    pass


@dataclass
class BoursoramaNews(BaseItem):
    source_name: str
    date: str
    news_title: str
    news_text: str
