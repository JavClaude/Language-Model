from dataclasses import dataclass


@dataclass
class BoursoramaEconomicNews:
    source_name: str
    date: str
    news_title: str
    news_text: str
