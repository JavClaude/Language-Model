from language_modeling.domain.scraper.pipelines.preprocess_pipeline import (
    CleanHtmlPipeline,
)

ITEM_PIPELINES = {CleanHtmlPipeline: 300}
