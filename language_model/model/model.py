import logging
from typing import Tuple, Union

from torch.cuda import is_available
from torch.nn import CrossEntropyLoss, Embedding, LayerNorm, Linear, LSTM, Module
from torch.optim import Adam
from torch import tensor, zeros
from tqdm import tqdm

from language_model.preprocessing.data import LanguageModelingDataset

device = "cuda" if is_available() else "cpu"

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)


class LstmModel(Module):
    """LstmModel is an object that performs next words prediction

    Parameters
    ----------
    vocabulary_size : int
        Vocabulary size that will be used for the embedding matrix
    embedding_dimension: int
        Dimension that will be used for the embedding matrix
    hidden_units: int
        Dimension that will be used in the lstm layers
    n_layers: int
        Number of lstm layers to use
    dropout_rnn: float
        dropout rate to be used in the lstm layers
    """
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_units: int,
        num_layers: int,
        dropout_rnn: float,
        **kwargs
    ) -> None:
        super(LstmModel, self).__init__()

        self.criterion = CrossEntropyLoss()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.embedding_layer = Embedding(vocabulary_size, embedding_dimension) # Put someting for padding index
        self.first_norrmalization_layer = LayerNorm(embedding_dimension)
        self.encoder_layer = LSTM(
            embedding_dimension, 
            hidden_units, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rnn
        )
        self.second_normalization_layer = LayerNorm(hidden_units)
        self.decoder_layer = Linear(hidden_units, vocabulary_size)
    
    def forward(self, inputs: Tuple[tensor, tensor]) -> Tuple[tensor, tensor]:
        """Forward pass of the LstmModel

        Parameters
        ----------
        inputs : Tuple[tensor]
            Tuple of tensor:
                - tensor of sequences
                - tensor of hidden states

        Returns
        -------
        Tuple[tensor, tensor]
            - logits tensor
            - hidden states
        """
        x = self.embedding_layer(inputs[0])
        x = self.first_norrmalization_layer(x)
        x, hidden_states = self.encoder_layer(x, inputs[1])
        x = self.second_normalization_layer(x)
        x = self.decoder_layer(x)

        return x, hidden_states
    
    def init_hidden(self, batch_size: int) -> Tuple[tensor, tensor]:
        return (
            zeros(size=(self.num_layers, batch_size, self.hidden_units), device=device),
            zeros(size=(self.num_layers, batch_size, self.hidden_units), device=device)
        )
    
    def fit(
        self, 
        train_data_iterator: LanguageModelingDataset, 
        eval_data_iterator: Union[LanguageModelingDataset, None] = None, 
        epochs: int = 1
    ) -> None:
        """Fit the object

        Parameters
        ----------
        train_data_iterator : LanguageModelingDataset
            Object fitted on data train
        eval_data_iterator : LanguageModelingDataset
            Object fitted on data test
        epochs : int
            Number of epochs to train the model for
        """
        self.to(device)
        self.zero_grad()

        logger.info(
            "Fitting model on {} samples".format(len(train_data_iterator))
        )

        self.optimizer = Adam(self.parameters())

        hidden_states = self.init_hidden(train_data_iterator.batch_size)
        
        for epoch in range(epochs):
            tmp_loss = 0
            self.train()
            for iteration in tqdm(range(0, len(train_data_iterator), train_data_iterator.bptt), desc="Training..."):
                batch = train_data_iterator.get_batches(iteration)
                batch = tuple(t.to(device) for t in batch)
                train_sequence, target_sequence = batch

                logits, hidden_states = self((train_sequence, hidden_states))
                hidden_states = tuple(t.detach() for t in hidden_states)

                loss = self.criterion(logits.transpose(2, 1), target_sequence)
                loss.backward()
                tmp_loss += loss.item()

                self.optimizer.step()
                self.zero_grad()
        
            logger.info(
                "Epoch: {}, Loss: {}".format(
                    epoch, tmp_loss / train_data_iterator.total_number_of_batches
                )
            )
            
            if eval_data_iterator is not None:
                self._evaluate(eval_data_iterator)
    
    def _evaluate(self, data_iterator: LanguageModelingDataset) -> None:
        self.to(device)
        self.eval()

        logger.info(
            "Evaluate model on {} samples".format(len(data_iterator))
        )

        hidden_states = self.init_hidden(data_iterator.batch_size)
        
        tmp_loss = 0
        for iteration in tqdm(range(0, len(data_iterator), data_iterator.bptt), desc="Evaluate..."):
            batch = data_iterator.get_batches(iteration)
            batch = tuple(t.to(device) for t in batch)
            train_sequence, target_sequence = batch

            logits, hidden_states = self((train_sequence, hidden_states))
            hidden_states = tuple(t.detach() for t in hidden_states)

            loss = self.criterion(logits.transpose(2, 1), target_sequence)
            tmp_loss += loss.item()

        logger.info(
            "Evaluation Loss: {}".format(
                tmp_loss / data_iterator.total_number_of_batches
            )
        )
