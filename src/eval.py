import argparse
import copy
import torch
import pathlib

from src.constants import MODELS_DIR
from src.corpus_loader import fetch_and_load_corpus
from src.model import RNN
from src.utils import one_hot_encoding

def load_model() -> RNN:
    filepath = pathlib.Path(f'{MODELS_DIR}/final_state_dict.pth')
    if not filepath.exists():
        raise RuntimeError("Not found pre-trained RNN model")
    return torch.load(filepath)

def generate(model: RNN, prompt: str, char_to_index: dict, index_to_char: dict, vocab_size: int, sequence_size: int = 16, output_size: int = 100) -> str:
    if len(prompt) < sequence_size:
        raise RuntimeError("Starting characters should have at least 16 symbols")
    
    embedding_tensor = one_hot_encoding(torch.tensor([char_to_index[char] for char in prompt[-sequence_size:]], dtype=torch.long), vocab_size).unsqueeze(0)
    chars = copy.deepcopy(prompt.split('\s'))

    model.eval()
    hidden_state = model.init_hidden()
    for _ in range(output_size - len(prompt)):
        with torch.no_grad():
            logits_tensor, hidden_state = model(embedding_tensor, hidden_state)
            logits_probs = torch.softmax(logits_tensor[-1], dim=-1).data

        char_idx = torch.multinomial(logits_probs, 1).item()
        char_tensor = one_hot_encoding(torch.tensor([char_idx]), vocab_size).unsqueeze(0)
        chars.append(index_to_char[char_idx])

        embedding_tensor = torch.cat((embedding_tensor[:, 1:, :], char_tensor), dim=1)

    return ''.join(chars)

def prompt(corpus: str, text: str, sequence_size: int, output_size: int):
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}
    index_to_char = {idx: char for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    model = load_model()

    return generate(model, text, char_to_index, index_to_char, vocab_size, sequence_size, output_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text based on a arbitrary corpus.")
    
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the corpus (e.g., Shakespeare corpus: "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt")')
    parser.add_argument('--prompt_text', type=str, required=True,
                        help='Text to use as a basis for text generation (e.g., "Forecasting for you")')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--output_size', type=int, required=True,
                        help='The size of the generated text output (e.g., "100")')

    args = parser.parse_args()

    corpus = fetch_and_load_corpus(args.url)
    
    print(prompt(corpus, args.prompt_text, args.sequence_size, args.output_size))