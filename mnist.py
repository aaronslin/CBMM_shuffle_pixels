from tensorflow.examples.tutorials.mnist import input_data
import nn_architecture as nn
import filename_paths

SIZE = (28, 28)
PADDED_SIZE = (32, 32)
CNN = nn.MNIST_Network

def init_om(isOM):
	global mnist
	mnist_path = filename_paths.get_mnist_data_path(isOM)
	mnist = input_data.read_data_sets(mnist_path, one_hot=True)

def get_next_batch(mode, batch_size):
	if mode == "train":
		return mnist.train.next_batch(batch_size)
	if mode == "test":
		x = mnist.test.images[:batch_size]
		y = mnist.test.labels[:batch_size]
		return x, y
