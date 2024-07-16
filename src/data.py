import numpy as np
import pathlib
import requests
import torch
import torch.nn.functional as F
import typing as t

def fetch_data(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise RuntimeError(f"Failed to fetch data. Status code: {response.status_code}, URL: {url}")

def fetch_and_save_corpus():
    filepath = pathlib.Path('dataset/shakespeare.txt')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        with open(filepath, 'r+') as f:
            text = f.read()
            return text
    text = fetch_data('https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt')
    with open(filepath, 'w+') as f:
        f.write(text)
        return text
    

def one_hot_encoding(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    one_hot_encoded = F.one_hot(indices, num_classes=vocab_size)
    return one_hot_encoded.float()

def corpus_to_dataloader(corpus: str, char_to_index: dict, vocab_size: int, sequence_size: int, batch_size: int, start_index: int = 0, end_index: t.Optional[int] = None):
    if end_index is None:
        end_index = len(corpus) - sequence_size

    corpus_indices = np.array([char_to_index[char] for char in corpus])
    
    sequences = [corpus_indices[i:i+sequence_size] for i in range(start_index, end_index)]
    targets = corpus_indices[start_index+1:end_index+1]

    sequences = np.array(sequences)
    targets = np.array(targets)

    total_sequences = end_index - start_index
    for i in range(0, total_sequences, batch_size):
        batch_sequences = sequences[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]
        if len(batch_sequences) == batch_size:
            yield one_hot_encoding(torch.tensor(batch_sequences, dtype=torch.long), vocab_size), torch.tensor(batch_targets, dtype=torch.long)