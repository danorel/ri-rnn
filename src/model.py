import torch
import torch.nn as nn
import typing as t

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.00)
    elif isinstance(m, RNNCell):
        torch.nn.init.xavier_uniform_(m.weight_xh)
        torch.nn.init.xavier_uniform_(m.weight_hh)
        if m.bias_xh is not None:
            m.bias_xh.data.fill_(0.00)
        if m.bias_hh is not None:
            m.bias_hh.data.fill_(0.00)

class RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int
    ):
        super(RNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_xh = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, requires_grad=False)
        x_gate = torch.matmul(x, self.weight_xh.T) + self.bias_xh
        h_gate = torch.matmul(h, self.weight_hh.T) + self.bias_hh
        return torch.tanh(x_gate + h_gate)

class RNN(nn.Module):
    def __init__(
        self,
        embedding_dimensions: int,
        hidden_size: int,
        vocab_size: int
    ):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.rnn_cell = RNNCell(
            input_size=embedding_dimensions,
            hidden_size=hidden_size
        )
        self.decoder = nn.Linear(
            in_features=hidden_size,
            out_features=vocab_size
        )

    def forward(self, x: torch.Tensor, h = None):
        batch_size, sequence_size, _ = x.size()
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device) 
        
        outputs = []
        for i in range(sequence_size):
            h = self.rnn_cell(x[:, i, :], h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=1)
        output = self.decoder(outputs[:, -1, :])
        return output, h
    
    def init_hidden(self, batch_size: t.Optional[int] = 1):
        return torch.zeros(batch_size, self.hidden_size)