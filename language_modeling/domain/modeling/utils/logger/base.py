from abc import abstractmethod


class BaseLogger:
    pass

    @abstractmethod
    def log_loss(self) -> None:
        pass

    @abstractmethod
    def log_perplexity(self) -> None:
        pass

    @abstractmethod
    def log_params(self) -> None:
        pass

    @abstractmethod
    def log_dir(self) -> None:
        pass
