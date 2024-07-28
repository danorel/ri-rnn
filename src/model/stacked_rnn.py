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

        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.layers.append(RNN(input_size, hidden_size, hidden_size, dropout))
        for _ in range(stack_size - 2):
            self.layers.append(RNN(hidden_size, hidden_size, hidden_size, dropout))
        self.layers.append(RNN(hidden_size, hidden_size, output_size, dropout))

    def forward(self, x: torch.Tensor, h = None):
        if h is None:
            h = self.init_hidden(x.size(0))

        o = x.clone()
        hn = torch.zeros_like(h)
        for layer, rnn in enumerate(self.layers):
            o, hi = rnn(o, h[layer])
            hn[layer] = hi

        return o, hn
    
    def init_hidden(self, batch_size: t.Optional[int] = 1):
        return torch.zeros(self.stack_size, batch_size, self.hidden_size)