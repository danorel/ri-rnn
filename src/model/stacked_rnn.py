import torch
import torch.nn as nn
import typing as t

from src.model.rnn import RNN

class StackedRNN(nn.Module):
    name = "stacked"

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.5,
        stack_size: int = 3,
    ):
        super(StackedRNN, self).__init__()

        self.name = "stacked_rnn"
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList([])
        self.layers.append(RNN(input_size, hidden_size, hidden_size, dropout))
        self.layers.extend([
            RNN(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(stack_size - 2)
        ])
        self.layers.append(RNN(hidden_size, hidden_size, output_size, dropout))

    def forward(self, x: torch.Tensor, hi = None):
        if hi is None:
            hi = self.init_hidden(x.size(0))

        o = x.clone()
        for layer, rnn in enumerate(self.layers):
            o, h = rnn(o, hi[layer])
            hi[layer] = h

        return o, hi
    
    def init_hidden(self, batch_size: t.Optional[int] = 1):
        last_layer = len(self.layers) - 1
        return [
            torch.zeros(batch_size, self.output_size if layer == last_layer else self.hidden_size)
            for layer in range(len(self.layers))
        ]