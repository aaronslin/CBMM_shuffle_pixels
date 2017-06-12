import numpy as np


# Generator functions
def pow2_dimensions(image):
	# Takes an input image (e.g. MNIST: 28 x 28)
	# Returns image with power-of-2 dimensions (32 x 32)

	pass

def generate_shuffle_map(logDim):
	pass



# Shuffle functions 
def _check_shuffle_map(map, logDim):
	# Given map, check that dimensions are (2^logDim) x (2^logDim)
	if map is None:
		return False
	pass

def _coord_map(paneSize, x,y, outMap, inMap):
	xq, xr = (x // paneSize, x % paneSize)
	yq, yr = (y // paneSize, y % paneSize)

	xq, yq = outMap[xq][yq]
	xr, yr = inMap[xr][yr]

	return (xq * paneSize + xr, yq * paneSize + yr)

def _shuffle_out(image, logDim, logPanes, outMap):


	# map a tuple in (range(32), range(32)) -> 
	pass

def _shuffle_in(image, logDim, logPanes, inMap):
	pass

def shuffle(image, logDim, logPanes, inShuffleMap=None, outShuffleMap=None):
	# Input: 32x32 image; 5 = log(32); 1; None, a 2x2 permutation image
	# Output: the 32x32 is split into four 16x16 panes
	# 		(four = (2^logPanes)^2)
	if _check_shuffle_map(outShuffleMap, logPanes):
		image = _shuffle_out(image, logDim, logPanes, outShuffleMap)

	if _check_shuffle_map(inShuffleMap, logDim - logPanes):
		image = _shuffle_in(image, logDim, logPanes, inShuffleMap)

	return image 


