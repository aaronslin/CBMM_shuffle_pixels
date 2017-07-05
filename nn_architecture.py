import tensorflow as tf

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')

# Architecture parameters

PARAMS = {
	"mnist_5x5_nopool": {"poolsize": 1},
	"mnist_5x5_pool": {}
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
		self.__dict__.update((k,v) for k,v in default.iteritems())
		self.__dict__.update((k,v) for k,v in kwargs.iteritems() if k in default)


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

