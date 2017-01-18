
import theano
import theano.tensor as T
import lasagne
import numpy as np
from os import path, listdir, makedirs
import sys
import argparse

import ThirtyMinNet as model
from load_data import load_image

class Bottleneck:

    def __init__(self, network, input_var, bottleneck_dir=''):

        out = lasagne.layers.get_output(network)

        self.function = theano.function([input_var], out)
        self.bottleneck_dir = bottleneck_dir

    def save_bottleneck(self, name, input_tensor):
        if not path.exists(self.bottleneck_dir):
            makedirs(self.bottleneck_dir)

        output = self.function(input_tensor)
        output_file = path.join(self.bottleneck_dir, name + '.npy')
        np.save(output_file, output)

    def load_bottleneck(self, name):
        input_file = path.join(self.bottleneck_dir, name + '.npy')
        return np.load(input_file)

    def load_all_bottlenecks(self):
        input_files = [path.join(self.bottleneck_dir, f) for f in listdir(self.bottleneck_dir) if f.endswith('.npy')]

        return map(lambda fpath: {"path": fpath, "data": np.load(fpath)}, input_files)

    def generate_bottlenecks(self, data_root, verbose=False):
        image_dir = path.join(data_root, 'img')
        image_files = [path.join(image_dir, f) for f in listdir(image_dir) if f.endswith('.jpg')]

        for image_file in image_files:
            image_array = load_image(image_file, flat=False)
            name = path.basename(image_file).split('.')[0]
            self.save_bottleneck(name, input_tensor=image_array)

            if verbose:
                print "Saving bottleneck for image %s" % name


def main(argv):
    argparser = argparse.ArgumentParser(description='Generate bottlenecks from the data directory')
    argparser.add_argument('-d', '--data-dir',
        dest='data_dir',
        help='Data root directory',
        required=True)
    argparser.add_argument('-b', '--bottleneck-dir',
        dest='bottle_dir',
        help='Folder to store the bottlenecks in',
        required=True)
    argparser.add_argument('-m', '--model-file',
        dest='model_file',
        help='.npz file containing the weights of JUST the convolutional layers of the neural network',
        required=True)

    args = argparser.parse_args(argv)

    input_var = T.tensor4('input')
    network = model.get_convolution_ops(dimensions=(1, 3, 723, 972), input_var=input_var)
    network = model.load_model(network, args.model_file)

    bottleneck = Bottleneck(network, input_var, args.bottle_dir)
    bottleneck.generate_bottlenecks(args.data_dir, verbose=True)


if __name__ == "__main__":
    main(sys.argv[1:])
