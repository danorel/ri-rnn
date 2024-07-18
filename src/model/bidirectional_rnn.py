import torch
import torch.nn as nn
import typing as t

from src.model.rnn import RNN

class BidirectionalRNN(nn.Module):
    name = "bidirectional"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5
    ):
        super(BidirectionalRNN, self).__init__()

        self.hidden_size = hidden_size

        self.forward_rnn = RNN(input_size, hidden_size, output_size, dropout)
        self.backward_rnn = RNN(input_size, hidden_size, output_size, dropout)

    def forward(self, x: torch.Tensor, h = None):
        if h is None:
            h = self.init_hidden(x.size(0)) 

        fo, fh = self.forward_rnn(x, h)
        bo, bh = self.backward_rnn(torch.flip(x, dims=[1]), h)

        h = fh + bh
        o = fo + bo

        return o, h
    
    def init_hidden(self, batch_size: t.Optional[int] = 1):
        return torch.zeros(batch_size, self.hidden_size)