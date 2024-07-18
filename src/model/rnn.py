import torch
import torch.nn as nn
import typing as t

from src.model.base import RNNCell

class RNN(nn.Module):
    name = "rnn"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5
    ):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn_cell = RNNCell(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.decoder = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h = None):
        if h is None:
            h = self.init_hidden(x.size(0))
        
        o = []
        for i in range(x.size(1)):
            h = self.rnn_cell(x[:, i, :], h)
            o.append(self.decoder(self.dropout(h)))
        o = torch.stack(o, dim=1)

        return o, h
    
    def init_hidden(self, batch_size: t.Optional[int] = 1):
        return torch.zeros(batch_size, self.hidden_size)