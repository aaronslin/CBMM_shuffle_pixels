import os

ROOT = {
	True: "/home/aaronlin",
	False: "."
}

def get_shuffle_maps_path(isOM):
	return os.path.join(ROOT[isOM], "shuffle_maps.npy")

def get_cifar_images_path(isOM):
	if isOM:
		return os.path.join(ROOT[isOM], "cifar10_data")
	return "/tmp/cifar10_data"

def get_mnist_data_path(isOM):
	return "/tmp/data/"
