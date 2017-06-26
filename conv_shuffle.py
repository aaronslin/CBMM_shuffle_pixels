'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import tensorflow as tf
import frame_shuffle
from itertools import product
import time
import argparse

def load_maps(filename):
	mapsDict = np.load(filename).item()
	return mapsDict


# Flags from frame_shuffle
THE_DATASET = "mnist"                       # "mnist", "cifar", "none"
LOGDIM = 5

# Varied parameters
def taskNum_to_params1(taskNum):
	# input:  taskNum from 0 to 19
	# output: (logPanes, hasOut, hasIn) \in (1-LOGDIM, 0-1, 0-1)
	logPanes = (taskNum % LOGDIM) + 1
	hasOut = (taskNum % 2) == 1
	hasIn = (taskNum % 4) > 1
	return (logPanes, hasOut, hasIn)

def get_filename_dir(isOM):
	if isOM:
		return "/home/aaronlin/shuffle_maps.npy"
	return "shuffle_maps.npy"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--slurm_task_num", default=0)
parser.add_argument("-o", "--openmind", default=1)
args = parser.parse_args()

taskNum = int(args.slurm_task_num)
taskParams = taskNum_to_params1(taskNum)
print("Parameters:", taskParams)

isOpenmind = int(args.openmind)
FILENAME_MAP = get_filename_dir(isOpenmind)
MAPS_DICT = load_maps(FILENAME_MAP)


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
image_len = 32

# Network Parameters
n_input = image_len * image_len # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, image_len, image_len, 1])

	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling
	conv1 = maxpool2d(conv1, k=2)

	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)

	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Store layers weight & bias
weights = {
	# 5x5 conv, 1 input, 32 outputs
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
	# 5x5 conv, 32 inputs, 64 outputs
	'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
	# fully connected, 7*7*64 inputs, 1024 outputs
	'wd1': tf.Variable(tf.random_normal([int(image_len/4*image_len/4)*64, 1024])),
	# 1024 inputs, 10 outputs (class prediction)
	'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([32])),
	'bc2': tf.Variable(tf.random_normal([64])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

def train_model(logPanes, hasOut, hasIn):
	def process_images(batch):
		return frame_shuffle.batch_shuffle(batch, \
						THE_DATASET, MAPS_DICT, logPanes, hasOut, hasIn)
	# Launch the graph
	acc = 0
	testAcc = 0
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			batch_x = process_images(batch_x)

			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
										   keep_prob: dropout})
			
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
																  y: batch_y,
																  keep_prob: 1.})
				testBatch_x = process_images(mnist.test.images[:256])
				testAcc = sess.run(accuracy, feed_dict={x: testBatch_x,
										  y: mnist.test.labels[:256],
										  keep_prob: 1.})
				print("Iter " + str(step*batch_size) + ", Training Accuracy= " + \
					  "{:.5f}".format(acc) + ", Test Acc.= "+ \
					  "{:.5f}".format(testAcc))
			step += 1
		print("Optimization Finished!")
		return acc, testAcc

def vary_parameters():
	bools = [True, False]
	for logPanes, hasOut, hasIn in product(range(1, LOGDIM), bools, bools):
		train_given_parameters((logPanes, hasOut, hasIn))		

def train_given_parameters(params = taskParams):
	(logPanes, hasOut, hasIn) = params
	print("#####################  NEW  ITERATION #####################")
	print("Parameters (logPanes, hasOut, hasIn): ", (logPanes, hasOut, hasIn))
	acc, testAcc = train_model(logPanes, hasOut, hasIn)
	print("\n(acc, testAcc):", (acc, testAcc))
	print("\n\n\n\n\n")

def view_shuffled_images():
	def process_images(batch, params):
		(logPanes, hasOut, hasIn) = params
		return frame_shuffle.batch_shuffle(batch, \
						THE_DATASET, MAPS_DICT, logPanes, hasOut, hasIn)
	from pixel_averaging import disp
	temp_batch_size = 1
	max_iters = 5
	bools = [True, False]

	for i in range(max_iters):
		batch_x, batch_y = mnist.train.next_batch(temp_batch_size)

		params = []
		for logPanes, hasOut, hasIn in product(range(1, LOGDIM), bools, bools):
			params.append((logPanes, hasOut, hasIn))
		modifs = [process_images(batch_x, param)[0].reshape((32, 32)) for param in params]
		disp(modifs, params)

if __name__ == "__main__":
	train_given_parameters()
	#view_shuffled_images()


