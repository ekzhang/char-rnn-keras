import os
import json
import argparse

import numpy as np

from model import build_model, save_weights

DATA_DIR = './data'
LOG_DIR = './logs'

BATCH_SIZE = 16
SEQ_LENGTH = 64

class TrainLogger(object):
    def __init__(self, file):
        self.file = os.path.join(LOG_DIR, file)
        self.epochs = 0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')

    def add_entry(self, loss, acc):
        self.epochs += 1
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)

def read_batches(T, vocab_size):
    length = T.shape[0]
    batch_chars = length / BATCH_SIZE

    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i]
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

def train(text, epochs=100, save_freq=10):
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)

    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH
    log = TrainLogger('training_log.csv')

    for epoch in range(epochs):
        print '\nEpoch {}/{}'.format(epoch + 1, epochs)
        
        losses, accs = [], []

        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            loss, acc = model.train_on_batch(X, Y)
            print 'Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc)
            losses.append(loss)
            accs.append(acc)

        log.add_entry(np.average(losses), np.average(accs))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print 'Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    train(open(os.path.join(DATA_DIR, args.input)).read(), args.epochs, args.freq)
