import os

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

MODEL_DIR = './model'

def save_epoch(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

def load_epoch(epoch):
    model = load_model(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))
    return model

def build_model(batch_size, seq_len, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    model = build_model(16, 64, 50)
    model.summary()
