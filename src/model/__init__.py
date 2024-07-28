import torch.nn as nn
import typing as t

from .bidirectional_rnn import BidirectionalRNN
from .rnn import RNN
from .stacked_rnn import StackedRNN

model_selector: t.Dict[str, nn.Module] = {
    model.name: model
    for model in [RNN, BidirectionalRNN, StackedRNN]
}