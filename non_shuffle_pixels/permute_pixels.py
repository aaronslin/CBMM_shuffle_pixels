import cv2
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Using random.shuffle
MNIST_DIMENSIONS = (28,28)
FIXED_ARRAY = False

perm784 = [573, 515, 748, 89, 42, 143, 368, 554, 141, 171, 13, 202, 441, 497, 728, 547, 572, 180, 340, 381, 58, 159, 209, 300, 470, 520, 703, 364, 605, 583, 78, 86, 274, 357, 161, 150, 776, 595, 624, 127, 491, 230, 122, 629, 217, 555, 580, 120, 586, 260, 183, 465, 527, 440, 2, 265, 403, 485, 747, 103, 334, 306, 534, 413, 746, 476, 387, 720, 656, 77, 576, 8, 244, 563, 542, 655, 168, 733, 87, 194, 640, 421, 492, 216, 503, 262, 740, 626, 82, 412, 510, 653, 231, 516, 477, 257, 277, 567, 276, 236, 571, 570, 291, 302, 380, 184, 44, 621, 724, 190, 646, 169, 164, 374, 45, 383, 19, 156, 197, 400, 651, 752, 562, 523, 660, 760, 309, 528, 688, 289, 170, 487, 416, 587, 91, 667, 266, 251, 644, 612, 179, 581, 672, 620, 240, 399, 112, 332, 455, 442, 218, 69, 530, 293, 524, 618, 408, 10, 730, 499, 189, 200, 135, 634, 698, 435, 41, 107, 214, 337, 606, 196, 243, 255, 333, 48, 355, 708, 137, 407, 108, 602, 560, 734, 473, 126, 478, 104, 129, 329, 566, 456, 253, 564, 402, 66, 454, 280, 193, 429, 447, 460, 568, 677, 97, 331, 119, 701, 252, 742, 213, 215, 637, 712, 256, 211, 227, 204, 711, 79, 687, 377, 593, 636, 474, 765, 369, 775, 697, 599, 133, 404, 669, 574, 371, 144, 356, 781, 113, 60, 261, 449, 206, 500, 512, 603, 258, 326, 585, 754, 537, 0, 417, 178, 128, 11, 365, 494, 154, 771, 367, 308, 162, 386, 36, 182, 21, 354, 9, 749, 321, 191, 7, 270, 767, 349, 466, 509, 70, 609, 704, 341, 233, 544, 425, 74, 548, 722, 659, 428, 385, 115, 268, 647, 682, 396, 469, 352, 715, 290, 132, 437, 444, 504, 232, 348, 219, 422, 393, 768, 539, 271, 294, 517, 234, 250, 372, 745, 780, 432, 39, 281, 174, 558, 600, 578, 49, 17, 686, 284, 285, 272, 538, 680, 1, 467, 324, 279, 376, 320, 397, 188, 47, 353, 80, 707, 525, 479, 511, 723, 452, 639, 662, 24, 384, 359, 310, 392, 100, 490, 303, 427, 614, 27, 613, 713, 109, 346, 648, 148, 508, 556, 702, 521, 304, 462, 772, 157, 181, 744, 714, 222, 394, 433, 223, 316, 241, 295, 275, 319, 674, 638, 94, 124, 186, 358, 264, 73, 344, 90, 770, 142, 493, 160, 678, 203, 628, 345, 679, 75, 63, 689, 482, 481, 67, 753, 242, 738, 212, 424, 26, 55, 350, 419, 85, 177, 522, 20, 105, 102, 363, 33, 726, 288, 106, 668, 158, 579, 4, 457, 298, 166, 575, 439, 259, 705, 235, 297, 25, 610, 51, 12, 68, 56, 228, 96, 118, 322, 15, 175, 756, 607, 136, 692, 700, 623, 59, 488, 32, 535, 519, 531, 111, 695, 220, 57, 401, 388, 71, 43, 92, 594, 187, 138, 763, 608, 757, 681, 155, 246, 167, 273, 448, 198, 633, 278, 415, 649, 758, 691, 185, 464, 28, 225, 343, 330, 22, 426, 88, 737, 533, 40, 616, 315, 147, 683, 484, 635, 378, 777, 165, 76, 671, 347, 684, 201, 305, 46, 430, 453, 592, 743, 596, 779, 627, 495, 630, 336, 287, 529, 762, 735, 650, 134, 461, 641, 434, 514, 411, 307, 395, 604, 766, 406, 591, 652, 665, 18, 409, 676, 643, 323, 30, 116, 622, 764, 541, 370, 459, 64, 50, 267, 675, 299, 263, 532, 498, 725, 207, 389, 53, 717, 282, 590, 131, 382, 545, 172, 339, 5, 52, 29, 145, 471, 391, 245, 130, 750, 62, 773, 642, 443, 472, 690, 123, 657, 248, 536, 249, 247, 486, 238, 351, 716, 645, 366, 114, 483, 301, 718, 696, 296, 283, 559, 195, 93, 445, 489, 577, 513, 98, 615, 549, 546, 54, 6, 694, 673, 619, 598, 438, 81, 72, 420, 759, 480, 84, 210, 706, 436, 65, 318, 152, 751, 597, 149, 736, 446, 390, 125, 286, 224, 502, 292, 463, 311, 173, 423, 338, 732, 699, 313, 99, 140, 14, 431, 23, 117, 163, 617, 205, 38, 335, 226, 589, 761, 328, 458, 565, 654, 755, 611, 663, 373, 61, 739, 327, 414, 601, 719, 360, 666, 727, 405, 774, 625, 37, 582, 269, 778, 151, 121, 526, 561, 239, 314, 110, 16, 632, 557, 783, 362, 709, 254, 685, 588, 237, 693, 518, 139, 312, 721, 83, 670, 496, 199, 507, 782, 741, 664, 101, 552, 584, 475, 553, 410, 506, 342, 631, 153, 540, 3, 501, 379, 31, 450, 550, 229, 317, 551, 569, 451, 176, 325, 769, 505, 375, 543, 729, 398, 208, 661, 95, 731, 361, 710, 221, 34, 192, 418, 146, 468, 658, 35]

def perm28(fixed=FIXED_ARRAY):
	array = [26, 19, 12, 8, 17, 11, 22, 23, 27, 24, 14, 2, 15, 10, 6, 0, 25, 3, 5, 20, 18, 4, 21, 16, 1, 9, 13, 7]
	if not fixed:
		random.shuffle(array)
	return array

def load_img(path):
	raw = cv2.imread(path, 0)
	return cv2.resize(raw, MNIST_DIMENSIONS)

def disp(images, name=0):
	if type(images) == type([]):
		print "hi"
		for img in images:
			cv2.imshow(str(name),img)
			name = name+1
	elif type(images) == type(np.array([])):
		print "byp"
		cv2.imshow(str(name), images)
	else:
		return
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def perm_1D(oldIMG, perm_func = perm28):
	(x,y) = oldIMG.shape
	newIMG = np.zeros((x,y))
	perm = perm_func()

	for i in range(x):
		for j in range(y):
			newIMG[i][j] = oldIMG[perm[i]][perm[j]]
	return newIMG

def permute_batch(batch):
	orig = batch[0].shape
	return np.array([perm_1D(img.reshape(MNIST_DIMENSIONS)).reshape(orig) for img in batch])



# digit1 = load_img("1_mnist.png")
# digit8 = load_img("8_mnist.png")

# dig8_new = perm_1D(digit8)
# disp([digit8, dig8_new])