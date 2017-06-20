import numpy as np
import unittest
import math
import random
from itertools import product

# Globals

DATASET_SIZES = {
	"mnist": (28, 28),
	"cifar": (32, 32)
}

# Generator functions
def _pow2(exponent):
	return int(math.pow(2, exponent))

def pow2_dimensions(image, pad_values=(0,0)):
	# Takes an input image (e.g. MNIST: 28 x 28)
	# Returns image with power-of-2 dimensions (32 x 32), and logDim
	if image.shape[0] != image.shape[1]:
		raise Exception("Error: Input image is not a square")
	original_n = image.shape[0]
	logDim = int(math.ceil(math.log(original_n, 2)))
	desired_n = _pow2(logDim)
	diff = desired_n - original_n

	pad = (diff//2, diff//2)
	if diff % 2 == 1:
		pad = ((diff-1)/2, (diff+1)/2)
	paddedImage = np.pad(image, pad, "constant", constant_values=pad_values)

	return paddedImage, logDim

def _generate_one_shuffle_map(logSideLen):
	dim = _pow2(logSideLen)
	coords = [(x,y) for x in range(dim) for y in range(dim)]
	random.shuffle(coords)

	return np.array(coords).reshape((dim, dim, -1))

def generate_both_maps(logDim, logPanes, hasOut, hasIn):
	if hasOut:
		outerMap = _generate_one_shuffle_map(logPanes)
	else:
		outerMap = _trivial_shuffle_map(logPanes)

	if hasIn:	
		innerMap = _generate_one_shuffle_map(logDim-logPanes)
	else:
		innerMap = _trivial_shuffle_map(logDim-logPanes)
	return outerMap, innerMap

# Shuffle functions 
def _coord_map(logPaneSize, x, y, outMap, inMap):
	paneSize = _pow2(logPaneSize)
	xq, xr = (x // paneSize, x % paneSize)
	yq, yr = (y // paneSize, y % paneSize)

	xq, yq = outMap[xq][yq]
	xr, yr = inMap[xr][yr]

	return (xq * paneSize + xr, yq * paneSize + yr)

def _trivial_shuffle_map(logDim):
	dim = _pow2(logDim)
	coordinates = [(x,y) for x in range(dim) for y in range(dim)]
	return np.array(coordinates).reshape((dim, dim, 2))

def _apply_image_map(image, map):
	# Given an nxn image and an nxnx2 mapping, return the permutation
	def apply_map(coord):
		return image[coord[0]][coord[1]]
	return np.apply_along_axis(apply_map, 2, map)

def shuffle(image, logDim, logPanes, outShuffleMap=None, inShuffleMap=None):
	# Input: 32x32 image; 5 = log(32); 1; None, a 2x2 permutation image
	# Output: the 32x32 is split into four 16x16 panes
	# 		(four = (2^logPanes)^2)
	dim = _pow2(logDim)
	logPaneSize = logDim - logPanes

	if image.shape[0]!=dim or image.shape[1]!=dim:
		raise Exception("ERROR: logDim and image.shape don't match")

	if outShuffleMap is None:
		outShuffleMap = _trivial_shuffle_map(logPanes)
	if inShuffleMap is None:
		inShuffleMap = _trivial_shuffle_map(logPaneSize)

	map = np.array([_coord_map(logPaneSize, x, y, outShuffleMap, inShuffleMap) \
			for x in range(dim) for y in range(dim)]).reshape((dim, dim, -1))
	
	newImage = _apply_image_map(image, map)
	return newImage 



# Batch processing methods
def batch_shuffle(batch, dataset, mapsDict, logPanes, hasOut, hasIn):
	image_dim = DATASET_SIZES[dataset]
	(outMap, inMap) = read_shuffle_maps(mapsDict, logPanes, hasOut, hasIn)
	resized = np.apply_along_axis(_reshape_pad_shuffle_unshape, 1, batch, \
					image_dim, logPanes, outMap, inMap)
	return resized

def _reshape_pad_shuffle_unshape(image, image_dim, logPanes, outMap, inMap):
	image = image.reshape(image_dim)
	image, logDim = pow2_dimensions(image)
	newImage = shuffle(image, logDim, logPanes, outMap, inMap)
	newImage = newImage.reshape((image.size,))
	return newImage

def save_shuffle_maps(filename, logDim):
	maps_dict = {}
	bools = [True, False]
	for logPanes, hasOut, hasIn in product(range(1, logDim), bools, bools):
		key = (logPanes, hasOut, hasIn)
		bothMaps = generate_both_maps(logDim, logPanes, hasOut, hasIn)
		maps_dict[key] = bothMaps
	np.save(filename, maps_dict)

def read_shuffle_maps(mapsDict, logPanes, hasOut, hasIn):	
	return mapsDict[(logPanes, hasOut, hasIn)]



# Unit Tests
class TestCoord(unittest.TestCase):
	def test_pow2_dimensions(self):
		n = 6
		m = 8
		image = np.arange(n*n).reshape((n,n))
		zeros = [[0]*m]

		desired = [[j+n*i for j in range(n)] for i in range(n)]
		desired = [[0]+sub+[0] for sub in desired]
		desired = np.array(zeros + desired + zeros).reshape((m,m))
		self.assertTrue((pow2_dimensions(image)[0] == desired).all())

	def test_generate_map_shape(self):
		logDim = 3
		map = _generate_one_shuffle_map(logDim)
		self.assertEqual(map.shape, (2**logDim, 2**logDim, 2))

	def test_coord_map_trivial(self):
		logPaneSize = 4
		(x,y) = (17,18)
		outMap = np.array([(0,0), (0,1), (1,0), (1,1)]).reshape((2,2,-1))
		inMap = np.array([[(i,j) for j in range(logPaneSize)] for i in range(logPaneSize)])
		self.assertEqual(_coord_map(logPaneSize, x, y, outMap, inMap), (x,y))

	def test_coord_map_out(self):
		logPaneSize = 4
		(x,y) = (17,18)
		outMap = np.array([(0,1), (1,0), (1,1), (0,0)]).reshape((2,2,-1))
		inMap = np.array([[(i,j) for j in range(logPaneSize)] for i in range(logPaneSize)])
		self.assertEqual(_coord_map(logPaneSize, x, y, outMap, inMap), (1,2))

	def test_image_map(self):
		dim = 4
		image = np.arange(dim*dim).reshape((dim, dim))
		map = np.array([(x,y) for x in range(dim) for y in range(dim)]).reshape((dim, dim, -1))
		equalMatrices = (image == _apply_image_map(image, map)).all()
		self.assertTrue(equalMatrices)

	def test_shuffle0(self):
		logDim = 2
		logPanes = 1
		dim = 2**logDim
		image = np.arange(dim*dim).reshape((dim,dim))
		equalMatrices = (image == shuffle(image, logDim, logPanes)).all()
		self.assertTrue(equalMatrices)

	def test_shuffle1(self):
		logDim = 2
		logPanes = 1
		dim = 2**logDim
		image = np.arange(dim*dim).reshape((dim,dim))
		outMap = np.array([[(0,0),(0,1)],[(1,1),(1,0)]])
		output = np.array(range(8) + [10,11,8,9,14,15,12,13]).reshape((dim,dim))
		equalMatrices = (output == shuffle(image, logDim, logPanes, outMap)).all()
		self.assertTrue(equalMatrices)

# Testing 
if __name__ == '__main__':
	unittest.main()










