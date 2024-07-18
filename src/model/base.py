import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.00)
    elif isinstance(m, RNNCell):
        torch.nn.init.xavier_uniform_(m.weight_xh)
        torch.nn.init.xavier_uniform_(m.weight_hh)
        torch.nn.init.zeros_(m.bias_xh)
        torch.nn.init.zeros_(m.bias_hh)

class RNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0
    ):
        super(RNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.weight_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_xh = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x, h = None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, requires_grad=False)
        x_gate = torch.matmul(x, self.weight_xh.T) + self.bias_xh
        h_gate = torch.matmul(h, self.weight_hh.T) + self.bias_hh
        return self.dropout(torch.tanh(x_gate + h_gate))