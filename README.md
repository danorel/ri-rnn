# ri-rnn

ri-rnn is a Python project which is dealing with RNN re-implementation. The idea is to understand the model details and how text generation can be trained via RNN.

## Installation & Contributing

Use the [DEVELOPER.md](./DEVELOPER.md) guide to run or contribute to the project.

## Usage

1. Train RNN agent on **default** shakespeare dataset with **default** hyperparameters:

```bash
python -m src.train
```

2. Train RNN agent on **custom** dataset (probably, it can be any .txt file) with **default** hyperparameters:

```bash
python -m src.train --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

3. Train RNN agent on **default** shakespeare dataset with the **custom** available and tunable hyperparameters: 'epochs', 'dropout', 'sequence_size', 'batch_size', 'learning_rate', and 'weight_decay':

```bash
python -m src.train --epochs 5 --sequence_size 32 --dropout 0.3 --batch_size 256 --learning_rate 0.0001 --weight_decay 0.0001
```

4. Evaluate RNN agent via generation sampling on **custom** dataset with **pre-trained** model hyperparameters:

```bash
python -m src.eval --url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --prompt_text 'Forecasting for you' --output_size 100
```

## License

[MIT](./LICENSE)
