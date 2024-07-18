from .bidirectional_rnn import BidirectionalRNN
from .rnn import RNN
from .stacked_rnn import StackedRNN

model_selector = {
    model.name: model
    for model in [RNN, BidirectionalRNN, StackedRNN]
}