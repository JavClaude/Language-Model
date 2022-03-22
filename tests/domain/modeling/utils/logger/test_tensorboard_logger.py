from unittest.mock import patch

from language_modeling.domain.modeling.utils.logger.tensorboard_logger import (
    TensorboardLogger,
)


# TO DO path the summary writer object / __init__, for each run of a test a folder is created

@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.SummaryWriter.add_scalar"
)
def test_tensorboard_logger_log_learning_should_call_the_summary_writer_add_scalar_method_with_correct_parameters(
    add_scalar_mock,
):
    # Given
    tensorboard_logger = TensorboardLogger()
    learning_rate = 0.0001
    iteration = 1
    tag_value = "learning rate"

    # When
    tensorboard_logger.log_loss(learning_rate, iteration, tag_value)

    # Then
    add_scalar_mock.assert_called_with(
        scalar_value=learning_rate, global_step=iteration, tag=tag_value
    )


@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.SummaryWriter.add_scalar"
)
def test_tensorboard_logger_log_loss_should_call_the_summary_writer_add_scalar_method_with_correct_parameters(
    add_scalar_mock,
):
    # Given
    tensorboard_logger = TensorboardLogger()
    loss = 0.1
    iteration = 1
    tag_value = "unit test loss"

    # When
    tensorboard_logger.log_loss(loss, iteration, tag_value)

    # Then
    add_scalar_mock.assert_called_with(
        scalar_value=loss, global_step=iteration, tag=tag_value
    )


@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.SummaryWriter.add_scalar"
)
def test_tensorboard_logger_log_perplexity_should_call_the_summary_writer_add_scalar_method_with_correct_parameters(
    add_scalar_mock,
):
    # Given
    tensorboard_logger = TensorboardLogger()
    perplexity = 0.1
    iteration = 1
    tag_value = "unit test perplexity"

    # When
    tensorboard_logger.log_perplexity(perplexity, iteration, tag_value)

    # Then
    add_scalar_mock.assert_called_with(
        scalar_value=perplexity, global_step=iteration, tag=tag_value
    )


@patch(
    "language_modeling.domain.modeling.utils.logger.tensorboard_logger.SummaryWriter.add_hparams"
)
def test_tensorboard_logger_log_params_should_call_the_summary_writer_add_hparams_method_with_correct_parameters(
    add_hparams_mock,
):
    # Given
    tensorboard_logger = TensorboardLogger()
    hparam_dict = {"a": 1, "b": 2}
    metric_dict = {"c": 1, "d": 2}

    # When
    tensorboard_logger.log_params(hparam_dict, metric_dict)

    # Then
    add_hparams_mock.assert_called_with(
        hparam_dict=hparam_dict, metric_dict=metric_dict
    )
