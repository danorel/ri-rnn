import copy
import torch
import pathlib

from src.data import fetch_and_save_corpus, one_hot_encoding
from src.model import RNN

def load_model(vocab_size):
    filepath = pathlib.Path('models/rnn_state_dict_0.pth')
    if not filepath.exists():
        raise RuntimeError("Not found pre-trained RNN model")

    hidden_size = vocab_size * 4

    model = RNN(
        embedding_dimensions=vocab_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size
    )
    model.load_state_dict(torch.load(filepath))

    return model

def generate(model: RNN, prompt: str, char_to_index: dict, index_to_char: dict, vocab_size: int, sequence_size: int = 16, output_size: int = 100) -> str:
    if len(prompt) < sequence_size:
        raise RuntimeError("Starting characters should have at least 16 symbols")
    
    embedding_tensor = one_hot_encoding(torch.tensor([char_to_index[char] for char in prompt[-sequence_size:]], dtype=torch.long), vocab_size).unsqueeze(0)
    chars = copy.deepcopy(prompt.split('\s'))

    model.eval()
    hidden_state = model.init_hidden()
    for _ in range(output_size - len(prompt)):
        logits_tensor, hidden_state = model(embedding_tensor, hidden_state)
        logits_probs = torch.softmax(logits_tensor[-1], dim=-1).data

        char_idx = torch.multinomial(logits_probs, 1).item()
        char_tensor = one_hot_encoding(torch.tensor([char_idx]), vocab_size).unsqueeze(0)
        chars.append(index_to_char[char_idx])

        embedding_tensor = torch.cat((embedding_tensor[:, 1:, :], char_tensor), dim=1)

    return ''.join(chars)

if __name__ == '__main__':
    corpus = fetch_and_save_corpus()
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    index_to_char = {idx: char for idx, char in enumerate(vocab)}

    prompt = 'Forecast for you: '
    vocab_size = len(vocab)
    sequence_size = 16
    output_size = 100
    model = load_model(vocab_size)

    print(generate(model, prompt, char_to_index, index_to_char, vocab_size, sequence_size, output_size))