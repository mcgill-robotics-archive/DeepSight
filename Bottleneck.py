
import theano
import lasagne
import numpy as np
from os import path, listdir


class Bottleneck:

    def __init__(self, network, input_var, bottleneck_dir=''):

        out = lasagne.layers.get_output(network)

        self.function = theano.function([input_var], out)
        self.bottleneck_dir = bottleneck_dir

    def save_bottleneck(self, name, input_tensor):
        output = self.function(input_tensor)
        output_file = path.join(self.bottleneck_dir, name + '.npy')
        np.save(output_file, output)

    def load_bottleneck(self, name):
        input_file = path.join(self.bottleneck_dir, name + '.npy')
        return np.load(input_file)

    def load_all_bottlenecks(self):
        input_files = [path.join(self.bottleneck_dir, f) for f in listdir(self.bottleneck_dir) if f.endswith('.npy')]

        return map(lambda fpath: {"path": fpath, "data": np.load(fpath)}, input_files)
