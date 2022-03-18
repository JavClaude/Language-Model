from typing import Union, Tuple

from numpy import ndarray
from torch import tensor, long

from language_modeling.domain.modeling import DEVICE


class LanguageModelingDataLoader:
    def __init__(
        self, bptt: int, sequence_of_ids: Union[ndarray, Tuple[ndarray, ndarray]]
    ) -> None:
        self.bptt = bptt
        self.sequence_of_ids = sequence_of_ids

    def __len__(self) -> int:
        if not self._check_if_sequence_of_ids_is_a_tuple(self.sequence_of_ids):
            return self.sequence_of_ids.shape[-1]
        return self.sequence_of_ids[0].shape[-1]

    def _check_if_sequence_of_ids_is_a_tuple(
        self, sequence_of_ids: Union[ndarray, Tuple[ndarray, ndarray]]
    ) -> bool:
        if isinstance(sequence_of_ids, tuple):
            self._check_sequence_of_ids_length(sequence_of_ids)
            return True
        return False

    @staticmethod
    def _check_sequence_of_ids_length(sequence_of_ids: Tuple[ndarray, ndarray]) -> None:
        if len(sequence_of_ids) > 2:
            raise ValueError(
                "Too many sequence detected, you must only provide a tuple of length 2"
            )

    def get_batches(self, batch_index: int) -> Union[tensor, Tuple[tensor, tensor]]:
        if self._check_if_sequence_of_ids_is_a_tuple(self.sequence_of_ids):
            yield (
                tensor(
                    self.sequence_of_ids[0][:, batch_index : batch_index + self.bptt],
                    device=DEVICE,
                    dtype=long,
                ),
                tensor(
                    self.sequence_of_ids[1][:, batch_index : batch_index + self.bptt],
                    device=DEVICE,
                    dtype=long,
                ),
            )
        yield tensor(
            self.sequence_of_ids[:, batch_index : batch_index + self.bptt],
            device=DEVICE,
            dtype=long,
        )
