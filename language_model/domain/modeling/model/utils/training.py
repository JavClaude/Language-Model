from typing import Tuple, Union

import torch
from torch import tensor
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from language_model.domain.modeling import DEVICE
from language_model.domain.modeling.data.dataloader import LanguageModelingDataLoader
from language_model.domain.modeling.model.neural_network.nn import LSTMModel


class Trainer:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.train_iteration = self._init_iteration()
        self.eval_iteration = self._init_iteration()

    @staticmethod
    def _init_iteration() -> int:
        return 0

    def train(
        self,
        model: LSTMModel,
        train_dataloader: LanguageModelingDataLoader,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
        eval_dataloader: Union[LanguageModelingDataLoader, None] = None,
        n_epochs: int = 3,
    ) -> None:
        self._put_model_on_the_device(model)
        for _ in range(n_epochs):
            self._train_on_epoch(model, train_dataloader, criterion, optimizer)
            if eval_dataloader:
                self._eval_on_epoch(model, eval_dataloader, criterion)

    @staticmethod
    def _put_model_on_the_device(model: LSTMModel) -> None:
        model.to(DEVICE)

    def _train_on_epoch(
        self,
        model: LSTMModel,
        train_dataloader: LanguageModelingDataLoader,
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
    ) -> None:
        self._clean_gradients(model)
        self._put_model_to_train_mode(model)
        hidden_states = model.init_hidden_states(self.batch_size)
        for batch_index in tqdm(
            range(0, len(train_dataloader), train_dataloader.bptt),
            desc="training model...",
        ):
            hidden_states = self._train_on_batch(
                model,
                train_dataloader.get_batches(batch_index),
                hidden_states,
                criterion,
                optimizer,
            )

    def _eval_on_epoch(
        self,
        model: LSTMModel,
        eval_dataloader: LanguageModelingDataLoader,
        criterion: CrossEntropyLoss,
    ) -> None:
        self._clean_gradients(model)
        self._put_model_to_eval_mode(model)
        with torch.no_grad():
            hidden_states = model.init_hidden_states(self.batch_size)
            for batch_index in tqdm(
                range(0, len(eval_dataloader), eval_dataloader.bptt),
                desc="evaluating model...",
            ):
                hidden_states = self._eval_on_batch(
                    model,
                    eval_dataloader.get_batches(batch_index),
                    hidden_states,
                    criterion,
                )

    @staticmethod
    def _clean_gradients(model: LSTMModel) -> None:
        model.zero_grad()

    @staticmethod
    def _put_model_to_train_mode(model: LSTMModel) -> None:
        model.train()

    @staticmethod
    def _put_model_to_eval_mode(model: LSTMModel) -> None:
        model.eval()

    def _train_on_batch(
        self,
        model: LSTMModel,
        sequences_of_ids: Tuple[tensor, tensor],
        hidden_states: Tuple[tensor, tensor],
        criterion: CrossEntropyLoss,
        optimizer: Optimizer,
    ) -> Tuple[tensor, tensor]:
        predictions, hidden_states = self._get_model_output(
            model, sequences_of_ids[0], hidden_states
        )
        hidden_states = self._detach_hidden_states(hidden_states)
        predictions = self._transpose_decoder_output_matrix(predictions)
        loss = self._compute_loss(criterion, predictions, sequences_of_ids[1])
        self._compute_gradients(loss)
        self._apply_gradient_descent(optimizer)
        self.train_iteration = self._increment_iteration(self.train_iteration)
        return hidden_states

    def _eval_on_batch(
        self,
        model: LSTMModel,
        sequence_of_ids: Tuple[tensor, tensor],
        hidden_states: Tuple[tensor, tensor],
        criterion: CrossEntropyLoss,
    ) -> Tuple[tensor, tensor]:
        predictions, hidden_states = self._get_model_output(
            model, sequence_of_ids[0], hidden_states
        )
        predictions = self._transpose_decoder_output_matrix(predictions)
        loss = self._compute_loss(criterion, predictions, sequence_of_ids[1])
        self.eval_iteration = self._increment_iteration(self.eval_iteration)
        return hidden_states

    @staticmethod
    def _get_model_output(
        model: LSTMModel, sequence_of_ids: tensor, hidden_states: Tuple[tensor, tensor]
    ) -> Tuple[tensor, tensor]:
        predictions, hidden_states = model(sequence_of_ids, hidden_states)
        return predictions, hidden_states

    @staticmethod
    def _detach_hidden_states(
        hidden_states: Tuple[tensor, tensor]
    ) -> Tuple[tensor, tensor]:
        return tuple(tensor.detach() for tensor in hidden_states)

    @staticmethod
    def _transpose_decoder_output_matrix(decoder_output_matrix: tensor) -> tensor:
        return decoder_output_matrix.transpose(2, 1)

    @staticmethod
    def _compute_loss(
        criterion: CrossEntropyLoss,
        decoder_output_matrix: tensor,
        target_sequence: tensor,
    ) -> tensor:
        return criterion(decoder_output_matrix, target_sequence)

    @staticmethod
    def _compute_gradients(loss: tensor) -> None:
        loss.backward()

    @staticmethod
    def _apply_gradient_descent(optimizer: Optimizer) -> None:
        optimizer.step()

    @staticmethod
    def _increment_iteration(iteration: int) -> int:
        return iteration + 1

    @staticmethod
    def _put_tensors_on_the_device(
        tensors: Tuple[tensor, tensor]
    ) -> Tuple[tensor, tensor]:
        return tuple(tensor.to(DEVICE) for tensor in tensors)
