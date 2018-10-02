# char-rnn-keras

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [Andrej Karpathy article](http://karpathy.github.io/2015/05/21/rnn-effectiveness) and original torch source code [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

## Requirements

This code was [initially](https://github.com/ekzhang/char-rnn-keras) written in Python 2 was ported to
Python 3 in this [current fork](https://github.com/TheErk/char-rnn-keras).
It requires the [Keras](https://keras.io) deep learning library.

## Input data

All input data should be placed in the [`data/`](./data) directory.

## Usage

To train the model with default settings:
```bash
$ python train.py --input=tiny-shakespeare.txt
```

To sample the model at epoch 100:
```bash
$ python sample.py 100
```

Training loss/accuracy is stored in `logs/training_log.csv`.
Model results including a graphical dump of the model and intermediate model weights are stored in `model` directory. This same directory is used by `sample.py` for loading the model after training.