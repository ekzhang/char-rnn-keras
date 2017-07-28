import argparse
import os
import json
import sys

import numpy as np

from model import build_model, load_weights

DATA_DIR = './data'

def sample(epoch, header, num_chars):
	with open(os.path.join(DATA_DIR, 'char_to_idx.json')) as f:
		char_to_idx = json.load(f)
	idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
	vocab_size = len(char_to_idx)

	model = build_model(1, 1, vocab_size)
	load_weights(epoch, model)

	sampled = [char_to_idx[c] for c in header]
	for c in header[:-1]:
		batch = np.zeros((1, 1, vocab_size))
		batch[0, 0, char_to_idx[c]] = 1
		model.predict_on_batch(batch)

	for i in range(num_chars):
		batch = np.zeros((1, 1, vocab_size))
		if sampled:
			batch[0, 0, sampled[-1]] = 1
		else:
			batch[0, 0, :] = np.ones(vocab_size) / vocab_size
		result = model.predict_on_batch(batch).ravel()
		sample = np.random.choice(range(vocab_size), p=result)
		sampled.append(sample)

	print ''.join(idx_to_char[c] for c in sampled)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Sample some text from the trained RNN.')
	parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
	parser.add_argument('--seed', default='', help='initial seed for the generated text')
	parser.add_argument('--len', type=int, default=512, help='number of characters to sample')
	args = parser.parse_args()

	print sample(args.epoch, args.seed, args.len)
