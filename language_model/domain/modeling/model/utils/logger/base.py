from abc import abstractmethod


class BaseLogger:
    pass

    @abstractmethod
    def log_loss(self) -> None:
        pass

    def log_perplexity(self) -> None:
        pass

