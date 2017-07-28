import os
import json
import sys

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

	def add_entry(self, hist):
		self.epochs += 1
		s = '{},{},{}\n'.format(self.epochs, hist['loss'][0], hist['acc'][0])
		with open(self.file, 'a') as f:
			f.write(s)

def read_batches(T, vocab_size):
	length = T.shape[0]
	batch_chars = length / BATCH_SIZE
	while True:
		for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
			X = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
			Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
			for batch_idx in range(0, BATCH_SIZE):
				for i in range(0, SEQ_LENGTH):
					c1 = T[batch_chars * batch_idx + start + i]
					c2 = T[batch_chars * batch_idx + start + i + 1]
					X[batch_idx, i, c1] = 1
					Y[batch_idx, i, c2] = 1
			yield X, Y

def train(text, epochs):
	char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
	with open(os.path.join(DATA_DIR, 'char_to_idx.json'), 'w') as f:
		json.dump(char_to_idx, f)

	idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
	vocab_size = len(char_to_idx)

	model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
	steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH
	log = TrainLogger('training_log.txt')

	for epoch in range(epochs):
		print 'Iteration {}/{}'.format(epoch + 1, epochs)
		hist = model.fit_generator(read_batches(T, vocab_size), steps_per_epoch)
		log.add_entry(hist.history)
		if (epoch + 1) % 10 == 0:
			save_weights(epoch + 1, model)

if __name__ == '__main__':
	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)

	file = sys.argv[1]
	model = train(open(os.path.join(DATA_DIR, file)).read(), 100)
