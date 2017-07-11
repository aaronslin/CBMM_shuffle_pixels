'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import tensorflow as tf
from itertools import product
import time
import argparse
import sys

def load_maps(filename):
	mapsDict = np.load(filename).item()
	return mapsDict


import frame_shuffle
import nn_architecture as nn
import filename_paths

# Flags from frame_shuffle
LOGDIM = 5

# Varied parameters
def taskNum_to_params1(taskNum):
	# input:  taskNum from 0 to 19
	# output: (logPanes, hasOut, hasIn) \in (1-LOGDIM, 0-1, 0-1)
	logPanes = (taskNum % LOGDIM) + 1
	hasOut = (taskNum % 2) == 1
	hasIn = (taskNum % 4) > 1
	return (logPanes, hasOut, hasIn)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--slurm_task_num", default=0, type=int)
parser.add_argument("-o", "--openmind", default=1, type=int)
parser.add_argument("-d", "--dataset", default=None)
parser.add_argument("-a", "--architecture", default="default")
args = parser.parse_args()

# Arg: Slurm task number
taskNum = args.slurm_task_num
taskParams = taskNum_to_params1(taskNum)
print("Parameters:", taskParams)

# Arg: Openmind usage
isOpenmind = args.openmind
FILENAME_MAP = filename_paths.get_shuffle_maps_path(isOpenmind)
MAPS_DICT = load_maps(FILENAME_MAP)

# Arg: Dataset name
DATASET_NAME = args.dataset
try:
	DATASET = __import__(DATASET_NAME)
except ImportError:
	print("Dataset not found:", DATASET_NAME)
	sys.exit(1)
DATASET.init_om(isOpenmind)

# Arg: CNN Architecture name
architecture_name = args.architecture

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
dropout = 0.75





# Construct model
convNet = DATASET.CNN(**nn.PARAMS[architecture_name])

x, y = convNet.get_placeholders()
keep_prob = tf.placeholder(tf.float32)
pred = convNet.predict(x, keep_prob)

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
						DATASET_NAME, MAPS_DICT, logPanes, hasOut, hasIn)
	# Launch the graph
	acc = 0
	testAcc = 0
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			batch_x, batch_y = DATASET.get_next_batch("train", batch_size)
			batch_x = process_images(batch_x)

			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
										   keep_prob: dropout})
			
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
																  y: batch_y,
																  keep_prob: 1.})
				test_x, test_y = DATASET.get_next_batch("test", batch_size)
				test_x = process_images(test_x)
				testAcc = sess.run(accuracy, feed_dict={x: test_x,
										  y: test_y,
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
						DATASET_NAME, MAPS_DICT, logPanes, hasOut, hasIn)
	from pixel_averaging import disp
	temp_batch_size = 1
	max_iters = 5
	bools = [True, False]

	for i in range(max_iters):
		batch_x, batch_y = DATASET.get_next_batch("train", temp_batch_size)

		params = []
		for logPanes, hasOut, hasIn in product(range(1, LOGDIM), bools, bools):
			params.append((logPanes, hasOut, hasIn))
		modifs = [process_images(batch_x, param)[0] \
					.reshape(DATASET.PADDED_SIZE) for param in params]
		disp(modifs, params)

if __name__ == "__main__":
	train_given_parameters()
	#view_shuffled_images()


