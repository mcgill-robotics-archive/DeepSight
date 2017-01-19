
from load_data import load_image, load_label, contains_buoy
from os import path, listdir
import numpy as np
import random


def _load_list(image_names, image_dir, label_dir, net_type):

    # TODO: This is where you would perform image resizing
    batch_images = map(lambda image_name: load_image(path.join(image_dir, image_name + '.jpg'), net_type=net_type), image_names)
    batch_labels = map(lambda image_name: contains_buoy(load_label(path.join(label_dir, image_name + '.txt'))), image_names)

    # Concatenate images and labels over batch dimension
    return np.concatenate(batch_images, axis=0), np.concatenate(batch_labels, axis=0)


class DataSet:

    def __init__(self, image_names, data_dir, net_type, batch_size = 1, image_size=None):
        # TODO: Implement image_size option so that if provided the images are resized after loaded.

        self.image_names = image_names
        self.data_dir = data_dir
        self.num_images = len(image_names)
        self.index = 0
        self.batch_size = batch_size
        self.image_size = image_size
        self.cache = None

        self.image_dir = path.join(self.data_dir, 'img')
        self.label_dir = path.join(self.data_dir, 'label')
        self.net_type = net_type
        if self.num_images % self.batch_size != 0:
            print "Warning: The number of images %d is not divisible by the batch size %d, some images will be ignored" \
                  % (self.num_images, self.batch_size)
            self.index = 0

    def get_epoch_steps(self):
        return int(self.num_images / self.batch_size)

    def set_batch_size(self, batch_size):
        # TODO: Make sure this isn't being done after the first batch is loaded
        self.batch_size = batch_size

        if self.num_images % self.batch_size != 0:
            print "Warning: The number of images %d is not divisible by the batch size %d, some images will be ignored" \
                  % (self.num_images, self.batch_size)
            self.index = 0

    # This method simply returns a images as an array of (BATCH, DEPTH, WIDTH, HEIGHT) and labels as an array (BATCH, LABEL)
    def next_batch(self):
        # TODO: Implement some kind of cache for images so we can optimize the training process.
        # If possible we should offer an option to cache images directly into a GPU tensorflow variable (if we do tensorflow gpu training)

        batch_names = self.image_names[self.index:self.index+self.batch_size]

        # Increment the index
        self.index += self.batch_size

        if self.index + self.batch_size >= self.num_images:
            self.index = 0

        return _load_list(batch_names, image_dir=self.image_dir, label_dir=self.label_dir, net_type = self.net_type)

    def load_all(self):

        return _load_list(self.image_names, image_dir=self.image_dir, label_dir=self.label_dir, net_type = self.net_type)


# Returns training sets, validation sets, testing sets in that order as a tuple
def create_data_sets(data_dir, training_reserve=0.7, testing_reserve=0.1, validation_reserve=0.2, net_type = "VGG"):

    # Make sure the proportions add to 1.0, if not warn the user and switch back to defaults
    if training_reserve + testing_reserve + validation_reserve != 1.0:
        print "Warning: Provided training, testing, validation reserves do not add to one, switching back to defaults."
        training_reserve = 0.7
        testing_reserve = 0.1
        validation_reserve = 0.2

    image_dir = path.join(data_dir, 'img')
    label_dir = path.join(data_dir, 'label')

    # Get the file name (without extension) of all the jpeg files in the image directory
    image_names = [path.basename(f).replace('.jpg', '') for f in listdir(image_dir) if f.endswith('.jpg')]

    # Filter out those without labels (for whatever reason)
    image_names = filter(lambda name: path.exists(path.join(label_dir, name + '.txt')), image_names)

    # Shuffle up the images
    random.shuffle(image_names)

    num_training = int(training_reserve * len(image_names))
    num_testing = int(testing_reserve * len(image_names))
    num_validation = int(validation_reserve * len(image_names))

    i = 0
    training_names = image_names[i:i+num_training]
    i += num_training
    testing_names = image_names[i:i+num_testing]
    i += num_testing
    validation_names = image_names[i:i+num_validation]

    return (DataSet(training_names, data_dir, net_type),
            DataSet(testing_names, data_dir, net_type),
            DataSet(validation_names, data_dir, net_type))


def resize_bulk(data_dir, img_size):
    image_dir = path.join(data_dir, 'img')
    files = listdir(image_dir)

    from scipy.misc import imread, imsave, imresize

    for image_file in files:
        print "Resizing %s" % path.basename(image_file)
        image_file = path.join(data_dir, 'img', image_file)
        im = imread(image_file, mode='RGB')
        im = imresize(im, img_size)
        imsave(image_file, im)
