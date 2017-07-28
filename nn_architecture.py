import tensorflow as tf
import numpy as np
import unittest
np.set_printoptions(threshold=np.nan)

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')

def conv_nolocality(x, W, n=32, channels=3, next_depth=64, k=3):
	# assuming 3x3 deconv; fixing the indices for now.
	# use tf.gather and tf.scatter_update
	index_shift = range(k*k)
	x_mat = tf.reshape(x, [-1, n*n*channels])
	

	W_shape = (n*n, k*k, channels, next_depth)
	W_matrix = _flat_scatter(index_shift, W, n)


	# W_shared = tf.cast(W_shared, tf.float64)
	# coords = tf.cast(_modulus_flat(index_shift, n), tf.int64)
	# indices = tf.cast(np.arange(n*n), tf.int64)

	# def deconv(indices):
	# 	#pixels = tf.gather(coords, index)
	# 	zeros = tf.Variable(np.zeros([n * n, channels, next_depth]), dtype=tf.float64)
	# 	mask = tf.scatter_update(zeros, indices, W_shared)
	# 	return mask
	# 	return tf.reshape(mask, [n*n*channels, next_depth])

	# W_mat = tf.map_fn(fn=deconv, elems=coords, dtype=tf.float64) 		# 1024 x (1024*3) x 64
	#W_mat = tf.reshape(W_mat, [n, n, (n*n*channels), next_depth])
	#W_mat = tf.transpose(W_mat, perm=[2, 0, 1, 3])

	return x_mat, W_matrix

def _flat_scatter(index_shift, W, n):
	'''
	Disclaimer: This code is /really/ ugly.
	
	The goal of this helper function is to use tf.scatter_update()
	to create the weight matrices that represent a nonlocal
	convolutions. The final weight matrix W_mat should have a shape: 
		(n*n, n*n, channels, next_depth)
	
	Dim[0]: We need to perform n*n convolutions, 1 centered around each
		pixel of the input image. Each slice along this dimension is
		a weight matrix representing each convolution. For example, if
		channels=next_depth=1 and n=4 with k(ernel_size)=2, we could have:

		(1 1 0 0 
		 1 1 0 0
		 0 0 0 0
		 0 0 0 0).flatten()

	Dim[1]: There are n*n pixels for a weight matrix slice that corresponds
		to a convolution centered around one pixel.
	Dim[2]: The number of channels in the image.
	Dim[3]: The depth of the next convolutional layer.


	The weight matrix should be sparse, so we start with a zero matrix,
	`zeros`. The variable `ind_flat` specifies which indices of the
	matrix `zeros` should be a convolution. The variable `W_flat`
	specifies these values. 

	All of these computations are flattened into (n*n*dim[0],) + dim[1:]
	arrays, so that tf.scatter_update is executed once instead of n*n 
	times. I couldn't figure out how to use tf.map_fn to call 
	tf.scatter_update n*n times, which would have been more elegant.
	'''

	(k, k, channels, next_depth) = W.shape

	# zSlice: 	(n*n, channels, next_depth) 	# 1 conv
	# zeros: 	(n*n*n*n, channels, next_depth) # n*n convs
	zSlice = np.zeros([n*n, channels, next_depth])
	zeros = np.concatenate([zSlice]* (n*n), axis=0) 

	# _flat, coords: 	(n*n, k*k)
	# ind_flat:			(n*n*k*k, )
	# Purpose: provide index offsets, since zeros is flattened
	coords = _modulus_flat(index_shift, n)
	_flat = np.hstack([np.arange(n*n).reshape((-1,1))] * (k*k))
	_flat = _flat * (n*n)
	ind_flat = (coords + _flat).flatten()

	# W_shared:	(k*k, channels, next_depth)
	# W_flat: 	(n*n*k*k, channels, next_depth)
	# Purpose: repeat values of W for all n*n convolutions
	W_shared = W.reshape((k*k, channels, next_depth))
	W_flat = np.concatenate([W_shared] * (n*n), axis=0)

	# orig_shape: (n*n, n*n, channels, next_depth)
	orig_shape = (n*n, n*n, channels, next_depth)
	out = tf.scatter_update(tf.Variable(zeros), ind_flat, W_flat)
	out = tf.reshape(out, orig_shape)
	return out

def _modulus_flat(index_shift, n):
	# Expecting index_shift to be array of numbers in [0, n^2-1]
	k_area = len(index_shift)
	shift = np.vstack([index_shift] * (n*n))
	index = np.vstack([np.arange(n*n)] * k_area).T
	# Shape: (n*n) x (k*k)
	ans = (index + shift) % (n*n)
	return ans

def get_deconv_indices(n, k, setting):
	'''
	Inputs:
		n: the side length of the image being convolved (32 for CIFAR-10)
		k: the side length of the kernel size
		setting: {"convolution", "consecutive", "random"}

	Returns: a length k*k array with relative indices to specify the
	pixels being convolved. The image is flattened into an n*n vector.
	For example, with a 6x6 image and a 3x3 convolution at the pixel
	marked X:

	0 0 0 0 0 0
	1 1 1 0 0 0
	1 X 1 0 0 0
	1 1 1 0 0 0 
	0 0 0 0 0 0 
	0 0 0 0 0 0 

	We should obtain the indices [-7, -6, -5, -1, 0, 1, 5, 6, 7], as
	these are the pixels (relative to the X) marked with a "1".

	Settings: 
		"convolution": See above example. A normal k*k convolution.
		"consecutive": For test purposes. Returns the next k*k pixels.
		"random": Returns a random set of k*k pixels.
	'''
	if setting == "convolution":
		min = -k // 2
		max = k // 2
		horiz = range(min+1, max+1)
		vert = [n*x for x in horiz]
		indices = [h+v for h in horiz for v in vert]
	if setting == "consecutive":
		indices = range(k*k)
	if setting == "random":
		pass
	return indices


class TF_Test(tf.test.TestCase):
	def test_nolocality(self):
		batch = 1
		n = 5
		channels = 3
		next_depth = 1
		k = 2
		x = np.arange(batch * n * n * channels).reshape((batch,n,n,channels))
		W = np.arange(1, k * k * channels * next_depth+1).reshape(k, k, channels, next_depth)
		#W = np.ones(k * k * channels * next_depth).reshape(k, k, channels, next_depth)

		xMat, WMat = conv_nolocality(x, W, n, channels, next_depth, k)
		init = tf.initialize_all_variables()
			
		with self.test_session() as sess:
			sess.run(init)
			xans, wans = sess.run([xMat, WMat])
			print(wans, wans.shape)


class Tests(unittest.TestCase):
	def test_modulus_flat(self):
		shifts = [0, 1, 3, 4]
		n = 3
		deconv = _modulus_flat(shifts, n)
		#print(deconv.shape)
		#print(deconv)



# Unit test

if __name__ == "__main__":
	#unittest.main()
	tf.test.main()


# Architecture parameters

PARAMS = {
	"default": {},
	"mnist_5x5_nopool": {"poolsize": 1},
	"mnist_5x5_pool": {},
	"cifar_pool1": {"d1": 32, "d2": 64, "d3": 384, "d4": 192}
}




# Network Object

class Network(object):
	def __init__(self):
		self.weights = None
		self.biases = None
		self.convnet = lambda x: NotImplemented

	def predict(self, x, keep_prob):
		return self.convnet(x, keep_prob)

class MNIST_Network(Network):
	def __init__(self, **kwargs):
		super(MNIST_Network, self).__init__()
		self.set_params(**kwargs)

		# MNIST settings
		self.image_len = 32
		self.n_input = self.image_len * self.image_len
		self.n_classes = 10

		# Initializing network architecture
		self.set_weights()
		self.set_biases()
		self.set_conv()

	def get_placeholders(self):
		x = tf.placeholder(tf.float32, [None, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		return x, y

	def set_params(self, **kwargs):
		default = {
			"d1": 32, 
			"d2": 64, 
			"d3": 1024, 
			"kernel": 5, 
			"poolsize": 2
		}
		self.__dict__.update((k,v) for k,v in default.items())
		self.__dict__.update((k,v) for k,v in kwargs.items() if k in default)


	def set_weights(self):
		width = self.image_len / (self.poolsize * self.poolsize)
		self.weights = {
			# 5x5 conv, 1 input, 32 outputs
			'wc1': tf.Variable(tf.random_normal([self.kernel, self.kernel, 1, self.d1])),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.random_normal([self.kernel, self.kernel, self.d1, self.d2])),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([int(width * width) * self.d2, self.d3])),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.d3, self.n_classes]))
		}

	def set_biases(self):
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([self.d1])),
			'bc2': tf.Variable(tf.random_normal([self.d2])),
			'bd1': tf.Variable(tf.random_normal([self.d3])),
			'out': tf.Variable(tf.random_normal([self.n_classes]))
		}

	def set_conv(self):
		weights = self.weights
		biases = self.biases

		def convnet(x, keep_prob):
			# Reshape input picture
			x = tf.reshape(x, shape=[-1, self.image_len, self.image_len, 1])

			# Convolution Layer
			conv1 = conv2d(x, weights['wc1'], biases['bc1'])
			# Max Pooling (down-sampling
			conv1 = maxpool2d(conv1, k=self.poolsize)

			# Convolution Layer
			conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
			# Max Pooling (down-sampling)
			conv2 = maxpool2d(conv2, k=self.poolsize)

			# Fully connected layer
			# Reshape conv2 output to fit fully connected layer input
			fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1)
			# Apply Dropout
			fc1 = tf.nn.dropout(fc1, keep_prob)

			# Output, class prediction
			out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
			return out
		self.convnet = convnet

class CIFAR_Network(Network):
	def __init__(self, **kwargs):
		super(CIFAR_Network, self).__init__()
		self.set_params(**kwargs)

		# MNIST settings
		self.image_len = 32
		self.n_input = self.image_len * self.image_len * 3
		self.n_classes = 10

		# Initializing network architecture
		self.set_weights()
		self.set_biases()
		self.set_conv()

	def predict(self, x, keep_prob):
		return self.convnet(x)

	def get_placeholders(self):
		x = tf.placeholder(tf.float32, [None, self.n_input])
		y = tf.placeholder(tf.float32, [None, self.n_classes])
		return x, y

	def set_params(self, **kwargs):
		default = {
			"d1": 32,
			"d2": 64,
			"d3": 384,
			"d4": 192,
			"kernel": 5,
			"poolsize": 2
		}
		self.__dict__.update((k,v) for k,v in default.items())
		self.__dict__.update((k,v) for k,v in kwargs.items() if k in default)


	def set_weights(self):
		width = self.image_len / (self.poolsize ** 3)
		self.weights = {
			'wc1': tf.Variable(tf.random_normal([self.kernel, self.kernel, 3, self.d1])),
			'wc2': tf.Variable(tf.random_normal([self.kernel, self.kernel, self.d1, self.d2])),
			'wc3': tf.Variable(tf.random_normal([self.kernel, self.kernel, self.d2, self.d3])),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.random_normal([int(width * width) * self.d3, self.d4])),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.random_normal([self.d4, self.n_classes]))
		}

	def set_biases(self):
		self.biases = {
			'bc1': tf.Variable(tf.random_normal([self.d1])),
			'bc2': tf.Variable(tf.random_normal([self.d2])),
			'bc3': tf.Variable(tf.random_normal([self.d3])),
			'bd1': tf.Variable(tf.random_normal([self.d4])),
			'out': tf.Variable(tf.random_normal([self.n_classes]))
		}

	def set_conv(self):
		weights = self.weights
		biases = self.biases

		def convnet(x):
			# Reshape input picture
			x = tf.reshape(x, shape=[-1, self.image_len, self.image_len, 3])
			print x

			# Convolution Layer
			conv1 = conv2d(x, weights['wc1'], biases['bc1'])
			conv1 = maxpool2d(conv1, k=self.poolsize)

			# Convolution Layer
			conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
			conv2 = maxpool2d(conv2, k=self.poolsize)

			# Convolution Layer
			conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
			conv3 = maxpool2d(conv3, k=self.poolsize)

			# Fully connected layer
			# Reshape conv3 output to fit fully connected layer input
			fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1)

			# Output, class prediction
			out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
			return out
		self.convnet = convnet

