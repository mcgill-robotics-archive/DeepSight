import numpy as np
from PIL import Image
from ast import literal_eval
import sys, glob

def load(path_to_data):
	#takes a string pointing to a location that contains an img folder with raw images, 
	#and a label folder that contains the labels and bounding boxes as text files for each frame
	files = glob.glob(path_to_data + '/img/*')
	images_w_labels = []

	for f in files:
		root = f[-9:-4]
		img = load_image(f, False)
		label = contains_buoy(load_label(path_to_data + '/label/' + root + '.txt'))
		images_w_labels.append([img,label])
	return images_w_labels

def contains_buoy(label):
	buoy = False
	for l in label:
		if l[1] == 1:
			buoy = True
	if buoy:
		label = [0,1]
	else:
		label = [1,0]
	return np.array(label).reshape((1, 2))

def load_label(f):
	file_obj = open(f,'rb')
	label = []
	for line in file_obj.readlines():
		label.append(literal_eval(line))
	return label

def load_image(f, flat=False):
	img = Image.open(f)
	img.load()
	data = np.asarray(img, dtype="float32")

	if flat:
		img_size = data.shape[0] * data.shape[1] * data.shape[2]
		data = img.reshape((1,img_size))
	else:
		# Add the batch dimension to the image
		data = data.reshape([1] + list(data.shape))
		# Set the dimension order to (BATCH, DEPTH, WIDTH, HEIGHT)

	return data

def main():
	print load(sys.argv[1])[0]
	# img = load_image(sys.argv[1], False)
	# print img
	

if __name__ == '__main__':
	main()
