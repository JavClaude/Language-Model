from unittest.mock import MagicMock, patch

from language_modeling.domain.modeling import DEVICE
from language_modeling.domain.modeling.utils.logger.tensorboard_logger import (
    TensorboardLogger,
)
from language_modeling.domain.modeling.utils.trainer import (
    TRAIN_LOSS_TAG,
    EVAL_LOSS_TAG,
)
from language_modeling.domain.modeling.utils.trainer.trainer import Trainer


def test_trainer_init_iteration_should_return_zero():
    # Given / When
    trainer = Trainer(1)
    expected = 0
    # Then
    assert trainer.train_iteration == expected
    assert trainer.eval_iteration == expected


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._put_model_on_the_device"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._train_on_epoch"
)
def test_trainer_train_method_should_call_train_on_epoch_n_times_when_n_epochs_is_equal_to_n(
    train_on_epoch_mock, put_model_on_the_device_mock
):
    # Given
    trainer = Trainer(1)
    model = "a"
    train_dataloader = "b"
    criterion = "c"
    optimizer = "d"
    n_epochs = 3

    # When
    trainer.train(model, train_dataloader, criterion, optimizer, None, n_epochs)

    # Then
    put_model_on_the_device_mock.assert_called()
    train_on_epoch_mock.assert_called_with(
        model, train_dataloader, criterion, optimizer
    )
    assert train_on_epoch_mock.call_count == n_epochs


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._put_model_on_the_device"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._train_on_epoch"
)
@patch("language_modeling.domain.modeling.utils.trainer.trainer.Trainer._eval_on_epoch")
def test_trainer_train_method_should_call_eval_on_epoch_n_times_when_n_epochs_is_equal_to_n(
    eval_on_epoch_mock, _, put_model_on_the_device_mock
):
    # Given
    trainer = Trainer(1)
    model = "a"
    train_dataloader = "b"
    criterion = "c"
    optimizer = "d"
    eval_dataloader = "e"
    n_epochs = 3

    # When
    trainer.train(
        model, train_dataloader, criterion, optimizer, eval_dataloader, n_epochs
    )

    # Then
    put_model_on_the_device_mock.assert_called()
    eval_on_epoch_mock.assert_called_with(model, eval_dataloader, criterion)
    assert eval_on_epoch_mock.call_count == n_epochs


def test_trainer_put_model_on_the_device_should_call_the_to_method_with_correct_device():
    # Given
    trainer = Trainer(1)
    lstm_model_mock = MagicMock()

    # When
    trainer._put_model_on_the_device(lstm_model_mock)

    # Then
    lstm_model_mock.to.assert_called_with(DEVICE)


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._clean_gradients"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._put_model_to_train_mode"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._train_on_batch",
    return_value="a",
)
def test_trainer_train_on_epoch_should_call_other_training_utils_methods(
    train_on_batch_mock, put_model_to_train_mode_mock, clean_gradients_mock
):
    # Given
    trainer = Trainer(1)

    model = MagicMock()
    init_hidden_states_return_value = "a"
    model.init_hidden_states.return_value = init_hidden_states_return_value

    train_dataloader = MagicMock()
    train_dataloader.bptt = 10
    train_dataloader.__len__.return_value = 100
    get_batches_return_value = yield 10
    train_dataloader.get_batches.return_value = get_batches_return_value

    criterion = "b"
    optimizer = "c"

    # When
    trainer._train_on_epoch(model, train_dataloader, criterion, optimizer)

    # Then
    clean_gradients_mock.assert_called_with(model)
    put_model_to_train_mode_mock.assert_called_with(model)
    model.init_hidden_states.assert_called_with(1)
    train_on_batch_mock.assert_called_with(
        model,
        get_batches_return_value,
        init_hidden_states_return_value,
        criterion,
        optimizer,
    )
    assert train_on_batch_mock.call_count == 10


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._clean_gradients"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._put_model_to_eval_mode"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._eval_on_batch",
    return_value="a",
)
def test_trainer_eval_on_epoch_should_call_other_eval_utils_methods(
    eval_on_batch_mock, put_model_to_eval_mode_mock, clean_gradients_mock
):
    # Given
    trainer = Trainer(1)

    model_mock = MagicMock()
    init_hidden_states_return_value = "a"
    model_mock.init_hidden_states.return_value = init_hidden_states_return_value

    eval_dataloader = MagicMock()
    eval_dataloader.bptt = 10
    eval_dataloader.__len__.return_value = 100
    get_batches_return_value = yield 10
    eval_dataloader.get_batches.return_value = get_batches_return_value

    criterion = "b"

    # When
    trainer._eval_on_epoch(model_mock, eval_dataloader, criterion)

    # Then
    clean_gradients_mock.assert_called_with(model_mock)
    put_model_to_eval_mode_mock.assert_called_with(model_mock)
    model_mock.init_hidden_states.assert_called_with(1)
    eval_on_batch_mock.assert_called_with(
        model_mock, get_batches_return_value, init_hidden_states_return_value, criterion
    )
    assert eval_on_batch_mock.call_count == 10


def test_trainer_clean_gradients_should_call_the_model_zero_grad_method():
    # Given
    trainer = Trainer(1)
    model_mock = MagicMock()

    # When
    trainer._clean_gradients(model_mock)

    # Then
    model_mock.zero_grad.assert_called()


def test_trainer_put_model_to_train_mode_should_call_the_model_train_method():
    # Given
    trainer = Trainer(1)
    model_mock = MagicMock()

    # When
    trainer._put_model_to_train_mode(model_mock)

    # Then
    model_mock.train.assert_called()


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._get_model_output",
    return_value=(1, 2),
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._detach_hidden_states",
    return_value=3,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._transpose_decoder_output_matrix",
    return_value=4,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_loss",
    return_value=5,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_gradients"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._apply_gradient_descent"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._increment_iteration"
)
def test_trainer_train_on_batch_should_call_other_training_utils_method(
    increment_iteration_mock,
    apply_gradient_descent_mock,
    compute_gradients_mock,
    compute_loss_mock,
    transpose_decoder_output_matrix_mock,
    detach_hidden_states_mock,
    get_model_output_mock,
):
    # Given
    trainer = Trainer(1)

    model = "a"
    hidden_states = "b"
    criterion = "c"
    optimizer = "d"

    first_sequence_value = 6
    second_sequence_value = 7
    sequence_of_ids = [first_sequence_value, second_sequence_value]

    # When
    _ = trainer._train_on_batch(
        model, sequence_of_ids, hidden_states, criterion, optimizer
    )

    # Then
    get_model_output_mock.assert_called_with(model, first_sequence_value, hidden_states)
    detach_hidden_states_mock.assert_called_with(2)
    transpose_decoder_output_matrix_mock.assert_called_with(1)
    compute_loss_mock.assert_called_with(criterion, 4, second_sequence_value)
    compute_gradients_mock.assert_called_with(5)
    apply_gradient_descent_mock.assert_called_with(optimizer)
    increment_iteration_mock.assert_called()


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._get_model_output",
    return_value=(1, 2),
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._transpose_decoder_output_matrix",
    return_value=3,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_loss",
    return_value=4,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._increment_iteration"
)
def test_trainer_eval_on_batch_should_call_other_training_utils_method(
    increment_iteration_mock,
    compute_loss_mock,
    transpose_decoder_output_matrix_mock,
    get_model_output_mock,
):
    # Given
    trainer = Trainer(1)

    model = "a"
    hidden_states = "b"
    criterion = "c"

    first_sequence_value = 6
    second_sequence_value = 7
    sequence_of_ids = [first_sequence_value, second_sequence_value]

    # When
    _ = trainer._eval_on_batch(model, sequence_of_ids, hidden_states, criterion)

    # Then
    get_model_output_mock.assert_called_with(model, first_sequence_value, hidden_states)
    compute_loss_mock.assert_called_with(criterion, 3, second_sequence_value)
    transpose_decoder_output_matrix_mock.assert_called_with(1)
    increment_iteration_mock.assert_called()


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._get_model_output",
    return_value=(1, 2),
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._detach_hidden_states",
    return_value=3,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._transpose_decoder_output_matrix",
    return_value=4,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_loss",
    return_value=5,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_gradients"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._apply_gradient_descent"
)
@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.TensorboardLogger.log_loss"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._increment_iteration",
    return_value=1,
)
def test_trainer_train_on_batch_should_call_other_training_utils_method_when_a_logger_is_setup(
    increment_iteration_mock,
    log_loss_mock,
    apply_gradient_descent_mock,
    compute_gradients_mock,
    compute_loss_mock,
    transpose_decoder_output_matrix_mock,
    detach_hidden_states_mock,
    get_model_output_mock,
):
    # Given
    trainer = Trainer(1)
    trainer.set_logger(TensorboardLogger())

    model = "a"
    hidden_states = "b"
    criterion = "c"
    optimizer = "d"

    first_sequence_value = 6
    second_sequence_value = 7
    sequence_of_ids = [first_sequence_value, second_sequence_value]

    # When
    _ = trainer._train_on_batch(
        model, sequence_of_ids, hidden_states, criterion, optimizer
    )

    # Then
    get_model_output_mock.assert_called_with(model, first_sequence_value, hidden_states)
    detach_hidden_states_mock.assert_called_with(2)
    transpose_decoder_output_matrix_mock.assert_called_with(1)
    compute_loss_mock.assert_called_with(criterion, 4, second_sequence_value)
    compute_gradients_mock.assert_called_with(5)
    apply_gradient_descent_mock.assert_called_with(optimizer)
    log_loss_mock.assert_called_with(5, 0, TRAIN_LOSS_TAG)
    increment_iteration_mock.assert_called()


@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._get_model_output",
    return_value=(1, 2),
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._transpose_decoder_output_matrix",
    return_value=3,
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._compute_loss",
    return_value=4,
)
@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.TensorboardLogger.log_loss"
)
@patch(
    "language_modeling.domain.modeling.utils.trainer.trainer.Trainer._increment_iteration"
)
def test_trainer_eval_on_batch_should_call_other_training_utils_method_when_a_logger_is_setup(
    increment_iteration_mock,
    log_loss_mock,
    compute_loss_mock,
    transpose_decoder_output_matrix_mock,
    get_model_output_mock,
):
    # Given
    trainer = Trainer(1)
    trainer.set_logger(TensorboardLogger())

    model = "a"
    hidden_states = "b"
    criterion = "c"

    first_sequence_value = 6
    second_sequence_value = 7
    sequence_of_ids = [first_sequence_value, second_sequence_value]

    # When
    _ = trainer._eval_on_batch(model, sequence_of_ids, hidden_states, criterion)

    # Then
    get_model_output_mock.assert_called_with(model, first_sequence_value, hidden_states)
    compute_loss_mock.assert_called_with(criterion, 3, second_sequence_value)
    transpose_decoder_output_matrix_mock.assert_called_with(1)
    log_loss_mock.assert_called_with(4, 0, EVAL_LOSS_TAG)
    increment_iteration_mock.assert_called()


def test_trainer_get_model_output_should_call_the_model_forward_pass():
    # Given
    trainer = Trainer(1)
    model_mock = MagicMock()
    model_mock.return_value = [1, 2]

    sequence_of_ids = [3, 4]
    hidden_states = [5, 6]

    # When
    _ = trainer._get_model_output(model_mock, sequence_of_ids, hidden_states)

    # Then
    model_mock.assert_called_with(sequence_of_ids, hidden_states)


def test_trainer_detach_hidden_states_should_call_the_detach_method_for_a_tuple_of_tensor():
    # Given
    trainer = Trainer(1)
    hidden_state_1 = MagicMock()
    hidden_state_2 = MagicMock()
    hidden_states = tuple((hidden_state_1, hidden_state_2))

    # When
    _ = trainer._detach_hidden_states(hidden_states)

    # Then
    hidden_state_1.detach.assert_called()
    hidden_state_2.detach.assert_called()


def test_trainer_transpose_decoder_output_matrix_should_call_the_transpose_method_with_correct_parameters():
    # Given
    trainer = Trainer(1)
    tensor_mock = MagicMock()

    # When
    _ = trainer._transpose_decoder_output_matrix(tensor_mock)

    # Then
    tensor_mock.transpose.assert_called_once_with(2, 1)


def test_trainer_compute_loss_should_call_the_forward_method_of_the_loss_module():
    # Given
    trainer = Trainer(1)
    criterion_mock = MagicMock()

    # When
    _ = trainer._compute_loss(criterion_mock, "a", "b")

    # Then
    criterion_mock.assert_called_with("a", "b")


def test_trainer_compute_gradients_should_call_the_backward_method_of_the_loss_tensor():
    # Given
    trainer = Trainer(1)
    loss_tensor_mock = MagicMock()

    # When
    trainer._compute_gradients(loss_tensor_mock)

    # Then
    loss_tensor_mock.backward.assert_called()


def test_trainer_apply_gradient_descent_should_call_the_optimizer_step_method():
    # Given
    trainer = Trainer(1)
    optimizer_mock = MagicMock()

    # When
    trainer._apply_gradient_descent(optimizer_mock)

    # Then
    optimizer_mock.step.assert_called()


def test_trainer_put_tensors_on_the_should_call_the_to_method_with_correct_device():
    # Given
    trainer = Trainer(1)
    tensor_1 = MagicMock()
    tensor_2 = MagicMock()
    tensors = tuple((tensor_1, tensor_2))

    # When
    _ = trainer._put_tensors_on_the_device(tensors)

    # Then
    tensor_1.to.assert_called_with(DEVICE)
    tensor_2.to.assert_called_with(DEVICE)
