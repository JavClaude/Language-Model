from typing import Dict

from torch.utils.tensorboard.writer import SummaryWriter

from language_modeling.domain.modeling.utils.logger.base import BaseLogger


class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir: str = None) -> None:
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_loss(self, loss: float, iteration: int, tag: str) -> None:
        self._writer.add_scalar(scalar_value=loss, global_step=iteration, tag=tag)

    def log_perplexity(self, perplexity: float, iteration: int, tag: str) -> None:
        self._writer.add_scalar(tag=tag, scalar_value=perplexity, global_step=iteration)

    def log_params(self, hparam_dict: Dict, metric_dict: Dict) -> None:
        self._writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    def log_dir(self) -> str:
        return self._writer.log_dir
