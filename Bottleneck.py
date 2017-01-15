
import theano
import lasagne
import numpy as np
from os import path


class Bottleneck:

    def __init__(self, network, input_var, data_dir=''):

        out = lasagne.layers.get_output(network)

        self.function = theano.function([input_var], out)
        self.data_dir = data_dir

    def save_bottleneck(self, name, input_tensor):
        output = self.function(input_tensor)
        output_file = path.join(self.data_dir, name)
        np.save(output_file, output)

    def load_bottleneck(self, name):
        input_file = path.join(self.data_dir, name)
        return np.load(input_file)

