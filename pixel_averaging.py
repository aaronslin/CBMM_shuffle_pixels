import numpy as np
import cv2

MNIST_DIMENSIONS = (28,28)
PIXEL_MAX_VALUE = 256

def load_img(path):
	raw = cv2.imread(path, 0)
	return cv2.resize(raw, MNIST_DIMENSIONS)

def disp(images, names=0):
	x_shift = 100
	y_shift = 25
	if type(images) == type([]):
		if names is 0:
			names = range(len(images))

		const_y = 25
		for (img, name) in zip(images, names):
			cv2.imshow(str(name),img)
			cv2.moveWindow(str(name), x_shift, y_shift)
			#cv2.moveWindow(str(name), x_shift + name[0]*200, \
			#			y_shift - 4*(name[0]-1)*(const_y + img.shape[1]))

			y_shift += img.shape[1] + const_y
	elif type(images) == type(np.array([])):
		cv2.imshow(str(names), images)
	else:
		return
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def generate_rand_grid(dimensions=MNIST_DIMENSIONS):
	return np.random.randint(PIXEL_MAX_VALUE, size=dimensions)

def modulus_shadow(image, hash):
	return (image-hash) % PIXEL_MAX_VALUE

def absolute_shadow(image, hash):
	return (image-hash)

def interleaved(image, hash, dimensions=MNIST_DIMENSIONS):
	(r,c) = dimensions
	shadow = absolute_shadow(image, hash)
	output = np.ravel(np.column_stack((hash, shadow))).reshape((-1,c))
	return output

def divorced(image, hash, dimensions=MNIST_DIMENSIONS):
	(r,c) = dimensions
	shadow = modulus_shadow(image, hash)
	output = np.row_stack((hash, shadow))
	return output

def batch_interleave(batch, hash):
	(x,y) = MNIST_DIMENSIONS
	out = [interleaved(img.reshape((x,y)), hash).reshape((2*x*y,)) \
				for img in batch]
	return np.array(out)

def batch_divorce(batch, hash):
	(x,y) = MNIST_DIMENSIONS
	out = [divorced(img.reshape((x,y)), hash).reshape((2*x*y,)) \
				for img in batch]
	return np.array(out)


# digit1 = load_img("1_mnist.png")
# digit8 = load_img("8_mnist.png")
