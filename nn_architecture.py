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

'''
TODO:
	- add option for biases
	- find a way to pad the image? (optional)
'''

def conv_nolocality(X, W_raw, setting):
	(batchSize, n, n, prevDepth) = X.shape
	(k, k, prevDepth, nextDepth) = W_raw.shape

	index_shift = _get_deconv_indices(n, k, setting)
	X = tf.reshape(X, (batchSize, n*n*prevDepth))
	W = _flat_scatter(index_shift, W_raw, n)

	X = tf.cast(X, tf.float32)
	W = tf.cast(W, tf.float32)

	Y = tf.matmul(X, W, b_is_sparse=True)
	Y = tf.reshape(Y, (batchSize, n*n, nextDepth))

	return X, W, Y

def _conv_matmul(x, W):
	'''
	Inputs:
		x: A (batchSize, n*n*prevDepth) shaped input batch
		W: A (n*n*prevDepth, n*n, nextDepth) shaped weight matrix
	Output:
		y: A (batchSize, n*n, nextDepth) shaped output matrix
	'''
	pass

def _flat_scatter(index_shift, W_raw, n):
	'''
	Disclaimer: This code is /really/ ugly.
	
	The goal of this helper function is to use tf.scatter_update()
	to create the weight matrices that represent a nonlocal
	convolutions. The final weight matrix W should have a shape: 
		(n*n, n*n, prevDepth, nextDepth)
	
	Dim[0]: We need to perform n*n convolutions, 1 centered around each
		pixel of the input image. Each slice along this dimension is
		a weight matrix representing each convolution. For example, if
		prevDepth=nextDepth=1 and n=4 with k(ernel_size)=2, we could have:

		(1 1 0 0 
		 1 1 0 0
		 0 0 0 0
		 0 0 0 0).flatten()

	Dim[1]: There are n*n pixels for a weight matrix slice that corresponds
		to a convolution centered around one pixel.
	Dim[2]: The number of channels in the image or the current layer.
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

	(k, k, prevDepth, nextDepth) = W_raw.shape

	# zSlice: 	(n*n, prevDepth, nextDepth) 	# 1 conv
	# zeros: 	(n*n*n*n, prevDepth, nextDepth) # n*n convs
	zSlice = np.zeros([n*n, prevDepth, nextDepth])
	zeros = np.concatenate([zSlice]* (n*n), axis=0) 

	# _flat, coords: 	(n*n, k*k)
	# ind_flat:			(n*n*k*k, )
	# Purpose: provide index offsets, since zeros is flattened
	coords = _modulus_flat(index_shift, n)
	_flat = np.hstack([np.arange(n*n).reshape((-1,1))] * (k*k))
	_flat = _flat * (n*n)
	ind_flat = (coords + _flat).flatten()

	# W_raw:	(k*k, prevDepth, nextDepth)
	# W_flat: 	(n*n*k*k, prevDepth, nextDepth)
	# Purpose: repeat values of W for all n*n convolutions
	W_raw = W_raw.reshape((k*k, prevDepth, nextDepth))
	W_flat = np.concatenate([W_raw] * (n*n), axis=0)

	# Shape after tf.scatter_update: (n*n*n*n, prevDepth, nextDepth)
	# 		after 2x tf.reshape: (n*n, n*n*prevDepth, nextDepth)
	# 		after tf.transpose: (n*n*prevDepth, n*n, nextDepth)
	# 		returned: (n*n*prevDepth, n*n*nextDepth)
	W = tf.scatter_update(tf.Variable(zeros), ind_flat, W_flat)
	W = tf.reshape(W, (n*n, n*n, prevDepth, nextDepth))
	W = tf.reshape(W, (n*n, n*n*prevDepth, nextDepth))
	W = tf.transpose(W, perm=[1, 0, 2])
	W = tf.reshape(W, (n*n*prevDepth, n*n*nextDepth))
	return W

def _modulus_flat(index_shift, n):
	# Expecting index_shift to be array of numbers in [0, n^2-1]
	k_area = len(index_shift)
	shift = np.vstack([index_shift] * (n*n))
	index = np.vstack([np.arange(n*n)] * k_area).T
	# Shape: (n*n) x (k*k)
	ans = (index + shift) % (n*n)
	return ans

def _get_deconv_indices(n, k, setting):
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
	def test_conv_arange(self):
		batch = 2
		n = 5
		prevDepth = 3
		nextDepth = 2
		k = 2
		x = np.arange(batch * n * n * prevDepth).reshape((batch,n,n,prevDepth))
		w = np.arange(1, k * k * prevDepth * nextDepth+1).reshape(k, k, prevDepth, nextDepth)
		setting = "convolution"

		X, W, Y = conv_nolocality(x, w, setting)
		init = tf.initialize_all_variables()
			
		with self.test_session() as sess:
			sess.run(init)
			xans, wans, yans = sess.run([X, W, Y])

	def test_conv_ones(self):
		x, w, setting, y = self.generate_xw_2()

		X, W, Y = conv_nolocality(x, w, setting)
		init = tf.initialize_all_variables()
			
		with self.test_session() as sess:
			sess.run(init)
			xans, wans, yans = sess.run([X, W, Y])
			print(xans)
			print("********************************")
			print(wans)
			print("********************************")
			print(yans)
			print(xans.shape, wans.shape, yans.shape)
			print("Expected:", y)

	def generate_xw_1(self):
		batch = 1
		n = 4
		prevDepth = 3
		nextDepth = 1
		k = 2

		rgb = range(1, prevDepth+1)
		x = np.array([100*i+j for i in range(n*n) for j in rgb])
		x = x.reshape((batch, n, n, prevDepth))

		ones = [1] * prevDepth
		w = np.concatenate([ones]*4, axis=0)
		w = w.reshape((k, k, prevDepth, nextDepth))

		setting = "consecutive"

		ybase = (24 + (np.arange(13)*2+3)*600).astype(np.float32)
		wrap_ind = [9, 6, 3]
		y = np.concatenate([ybase, ybase[wrap_ind]], axis=0).reshape((batch, -1, nextDepth))

		return x, w, setting, y
		
	def generate_xw_2(self):
		batch = 3
		x, w, setting, y = self.generate_xw_1()

		x = np.vstack([x]*batch)
		y = np.vstack([y]*batch)

		return x, w, setting, y


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

