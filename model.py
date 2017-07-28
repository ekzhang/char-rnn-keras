import os

import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation

MODEL_DIR = './model'

def save_weights(epoch, model):
	if not os.path.exists(MODEL_DIR):
		os.makedirs(MODEL_DIR)
	model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))

def load_weights(epoch, model):
	model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))
	return model

def build_model(batch_size, seq_len, vocab_size):
	model = Sequential()
	model.add(LSTM(256, return_sequences=True, batch_input_shape=(batch_size, seq_len, vocab_size), stateful=True))
	model.add(Dropout(0.5))
	for i in range(2):
		model.add(LSTM(256, return_sequences=True, stateful=True))
		model.add(Dropout(0.5))

	model.add(TimeDistributed(Dense(vocab_size)))
	model.add(Activation('softmax'))
	return model

if __name__ == '__main__':
	model = build_model(BATCH_SIZE, SEQ_LENGTH, 50)
	model.summary()
