import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.data import fetch_and_save_corpus, corpus_to_dataloader
from src.model import RNN, init_weights

if __name__ == '__main__':
    corpus = fetch_and_save_corpus()
    vocab = sorted(set(corpus))

    char_to_index = {char: idx for idx, char in enumerate(vocab)}

    epochs = 3
    sequence_size = 16
    batch_size = 512
    vocab_size = len(vocab)
    hidden_size = vocab_size * 4

    model = RNN(
        embedding_dimensions=vocab_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size
    )
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Training started...")
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        hidden_state = model.init_hidden(batch_size)

        for batch, (embedding_batch_tensor, target_batch_tensor) in enumerate(corpus_to_dataloader(corpus, char_to_index, vocab_size, sequence_size, batch_size)):
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
        torch.save(model.state_dict(), f'models/rnn_state_dict_{epoch}.pth')

    torch.save(model.state_dict(), 'models/rnn_state_dict.pth')
    print("Training completed.")