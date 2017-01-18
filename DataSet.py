
from load_data import load_image, load_label, contains_buoy
from os import path
import numpy as np

def _load_list(image_names, image_dir, label_dir):
    batch_images = map(lambda image_name: load_image(path.join(image_dir, image_name + '.jpg')), image_names)
    batch_labels = map(lambda image_name: contains_buoy(load_label(path.join(label_dir, image_name + '.txt'))), image_names)

    # Concatenate images and labels over batch dimension
    return np.concatenate(batch_images, axis=0), np.concatenate(batch_labels, axis=0)


class DataSet:

    def __init__(self, image_names, data_dir, batch_size = 1, image_size=None):
        self.image_names = image_names
        self.data_dir = data_dir
        self.num_images = len(image_names)
        self.index = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.cache = None

        self.image_dir = path.join(self.data_dir, 'img')
        self.label_dir = path.join(self.data_dir, 'label')

    def get_epoch_steps(self):
        return int(self.num_images / self.batch_size)

    # This method simply returns a images as an array of (BATCH, DEPTH, WIDTH, HEIGHT) and labels as an array (BATCH, LABEL)
    def next_batch(self):
        # Todo: Implement some kind of cache for images so we can optimize the training process.
        # If possible we should offer an option to cache images directly into a GPU tensorflow variable (if we do tensorflow gpu training)

        if self.index + self.batch_size > self.num_images:
            print "Warning: The number of images is not divisible by the batch size, some images will be ignored"
            self.index = 0

        batch_names = self.image_names[self.index:self.index+self.batch_size]
        self.index += self.batch_size

        if self.index + self.batch_size >= self.num_images:
            self.index = 0

        return _load_list(batch_names, image_dir=self.image_dir, label_dir=self.label_dir)


    def load_all(self):

        return _load_list(self.image_names, image_dir=self.image_dir, label_dir=self.label_dir)
