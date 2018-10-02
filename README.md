# char-rnn-keras

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [Andrej Karpathy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness) and the original torch source code [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

## Requirements

This code is written in Python 3, and it requires the [Keras](https://keras.io) deep learning library.

## Input data

All input data should be placed in the [`data/`](./data) directory. Sample training texts are provided.

## Usage

To train the model with default settings:
```bash
$ python train.py --input tiny-shakespeare.txt
```

To sample the model at epoch 100:
```bash
$ python sample.py 100
```

Training loss/accuracy is stored in `logs/training_log.csv`. Model results, including a graphical dump of the model and intermediate model weights, are stored in the `model` directory. This same directory is used by `sample.py` for loading the model after training.
