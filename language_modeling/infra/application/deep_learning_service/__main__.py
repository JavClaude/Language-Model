import os

import uvicorn

from language_modeling.infra.application.deep_learning_service.service.service import DeepLearningService
from language_modeling.infra.application.deep_learning_service.utils.load_artifacts import ArtifactsLoader


DEEP_LEARNING_SERVICE_PORT = os.environ.get("DEEP_LEARNING_SERVICE_PORT")
PATH_TO_DEEP_LEARNING_ARTIFACTS = os.environ.get("PATH_TO_DEEP_LEARNING_ARTIFACTS")


def main():
    artifacts_loader = ArtifactsLoader()
    preprocessor, model = artifacts_loader.load_preprocessor_and_model(PATH_TO_DEEP_LEARNING_ARTIFACTS)
    deep_learning_service = DeepLearningService(tokenizer=preprocessor._tokenizer, model=model)
    deep_learning_service.build_api()
    uvicorn.run(deep_learning_service.get_api(), host="0.0.0.0", port=DEEP_LEARNING_SERVICE_PORT, log_level="info")
