import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMModel(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                embedding_dim: int,
                hidden_units: int,
                n_layers: int,
                bidirectional: bool,
                dropout_rnn: float,
                dropout: float,
                **kwargs):
        super(LSTMModel, self).__init__()

        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.bidirectional = bidirectional

        self.Embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.D1 = torch.nn.Dropout(dropout)
        self.LN = torch.nn.LayerNorm()
        self.Encoder = torch.nn.LSTM(embedding_dim, hidden_units, num_layers=n_layers, batch_first=True, dropout=dropout_rnn, bidirectional=bidirectional)
        self.LN2 = torch.nn.LayerNorm()
        self.D2 = torch.nn.Dropout(dropout)
        self.Decoder = torch.nn.Linear(hidden_units*2 if bidirectional else hidden_units, vocab_size)
    
    def forward(self, sequence, hiddens):
        x = self.Embedding(sequence)
        x = self.D1(x)
        x = self.LN(x)
        x, hiddens = self.Encoder(x, hiddens)
        x = self.LN2(x)
        x = self.D2(x)
        logits = self.Decoder(x)
        return logits, hiddens
    
    def init_hiddens(self, batch_size):
        return (
            torch.zeros(size=(self.n_layers * 2 if self.bidirectional else self.n_layers, batch_size, self.hidden_units), device=device),
            torch.zeros(size=(self.n_layers * 2 if self.bidirectional else self.n_layers, batch_size, self.hidden_units), device=device)
        )
