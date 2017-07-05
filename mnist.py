from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def get_next_batch(mode, batch_size):
	if mode == "train":
		return mnist.train.next_batch(batch_size)
	if mode == "test":
		x = mnist.test.images[:batch_size]
		y = mnist.test.labels[:batch_size]
		return x, y