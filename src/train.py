import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pathlib

from tqdm import tqdm

from src.constants import MODELS_DIR
from src.corpus_loader import fetch_and_load_corpus
from src.data_loader import dataloader
from src.model import RNN, init_weights


def train(corpus: str, epochs: int, dropout: float, sequence_size: int, batch_size: int, learning_rate: float, weight_decay: float):
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    hidden_size = vocab_size * 4

    model = RNN(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
        dropout=dropout
    )
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model_dir = pathlib.Path(MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        hidden_state = model.init_hidden(batch_size)

        for batch, (embedding_batch_tensor, target_batch_tensor) in enumerate(dataloader(corpus, char_to_index, vocab_size, sequence_size, batch_size)):
            logits_batch_tensor, hidden_state = model(embedding_batch_tensor, hidden_state)
            hidden_state = hidden_state.detach()
            
            batch_loss = criterion(logits_batch_tensor, target_batch_tensor)
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_loss = batch_loss.item()
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f"Epoch {epoch+1}/{3}, Batch Loss (batch = {batch}): {batch_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{5}, Total Loss: {total_loss:.4f}")
        torch.save(model, model_dir / f'{epoch}_state_dict.pth')

    torch.save(model, model_dir / 'final_state_dict.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a RNN model on a specified text corpus.")
    
    parser.add_argument('--url', type=str, default='https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt',
                        help='URL to fetch the text corpus')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for training')
    parser.add_argument('--sequence_size', type=int, default=16,
                        help='The size of each input sequence')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of samples in each batch')
    parser.add_argument('--learning_rate', type=int, default=0.001,
                        help='Learning rate parameter of RNN optimizer (Adam is a default setting)')
    parser.add_argument('--weight_decay', type=int, default=0.001,
                        help='Weight decay parameter of RNN optimizer (Adam is a default setting) which serves for weights normalization to avoid overfitting')

    args = parser.parse_args()

    corpus = fetch_and_load_corpus(args.url)
    
    train(corpus, args.epochs, args.dropout, args.sequence_size, args.batch_size, args.learning_rate, args.weight_decay)