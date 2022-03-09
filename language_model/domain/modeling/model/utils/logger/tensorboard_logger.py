from torch.utils.tensorboard.writer import SummaryWriter

from language_model.domain.modeling.model.utils.logger.base import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: str = None) -> None:
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_loss(self, loss: float, iteration: int, tag: str) -> None:
        self._writer.add_scalar(
            tag=tag,
            scalar_value=loss,
            global_step=iteration
        )

    def log_perplexity(self, perplexity: float, iteration: int, tag: str) -> None:
        self._writer.add_scalar(
            tag=tag,
            scalar_value=perplexity,
            global_step=iteration
        )