import numpy as np
from itertools import cycle
from pixel_averaging import disp

np.set_printoptions(threshold='nan')
IMAGES = {}
LABELS = {}

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def prepare_cifar(mode):
	import os
	data_dir = "/tmp/cifar10_data"

	if mode == "train":
		num_datafiles = 5
		filenames = [os.path.join(data_dir, 'data_batch_%d' % i)
					for i in xrange(1, num_datafiles+1)]
		files = [unpickle(file) for file in filenames]
		images = np.concatenate([file["data"] for file in files])
		labels = np.concatenate([file["labels"] for file in files])
	if mode == "test":
		testFile = unpickle(os.path.join(data_dir, 'test_batch'))
		images = testFile["data"]
		labels = testFile["labels"]

	return cycle(images), cycle(labels)

def bgr_ify(image):
	interm = image.reshape((3, 32, 32))
	interm = interm[::-1]		# converts rgb -> bgr
	bgr = np.swapaxes(np.swapaxes(interm, 0, 2), 0, 1)
	return bgr

def grayscale_flat(image):
	bgr = bgr_ify(image)
	gray = np.dot(bgr, [0.114, 0.587, 0.299]) / 256
	return gray.reshape((-1,))

def one_hot(label, num_classes=10):
	return (np.arange(num_classes) == label).astype(np.int32)

def get_next_batch(mode, batch_size):
	X = []
	Y = []
	for _ in range(batch_size):
		image = grayscale_flat(next(IMAGES[mode]))
		label = one_hot(next(LABELS[mode]))
		X.append(image)
		Y.append(label)
	return np.array(X), np.array(Y)


for mode in ["train", "test"]:
	IMAGES[mode], LABELS[mode] = prepare_cifar(mode)


