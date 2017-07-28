# char-rnn-keras

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

### Requirements

This code is written in Python 2, using the [Keras](https://keras.io) deep learning library. It also requires [HDF5](https://support.hdfgroup.org/HDF5/) and [h5py](http://www.h5py.org).

### Usage

All input data should be placed in the `data/` directory. The example `jigs.txt` is taken from the [Nottingham Music Database](http://abc.sourceforge.net/NMD/).

To train the model (checkpoints by default every 10 epochs):
```bash
$ python train.py jigs.txt
```

To sample the model:
```bash
$ python sample.py 100 --seed X: --len 512
```

Training loss/accuracy is stored in `logs/training_log.txt`.
