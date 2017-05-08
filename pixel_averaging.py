import numpy as np
import cv2

MNIST_DIMENSIONS = (28,28)
PIXEL_MAX_VALUE = 256

def load_img(path):
	raw = cv2.imread(path, 0)
	return cv2.resize(raw, MNIST_DIMENSIONS)

def disp(images, name=0):
	if type(images) == type([]):
		for img in images:
			cv2.imshow(str(name),img)
			name = name+1
	elif type(images) == type(np.array([])):
		cv2.imshow(str(name), images)
	else:
		return
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def generate_rand_grid(dimensions=MNIST_DIMENSIONS):
	return np.random.randint(PIXEL_MAX_VALUE, size=dimensions)

def modulus_shadow(image, hash):
	return (image-hash) % PIXEL_MAX_VALUE

def interleaved(image, hash, dimensions=MNIST_DIMENSIONS):
	(r,c) = dimensions
	shadow = modulus_shadow(image, hash)
	output = np.ravel(np.column_stack((hash, shadow))).reshape((-1,c))
	return output

def divorced(image, hash, dimensions=MNIST_DIMENSIONS):
	(r,c) = dimensions
	shadow = modulus_shadow(image, hash)
	output = np.row_stack((hash, shadow))
	return output


digit1 = load_img("1_mnist.png")
digit8 = load_img("8_mnist.png")
