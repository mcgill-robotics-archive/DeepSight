import numpy
import theano
import theano.tensor as T
import lasagne

from collections import OrderedDict
import pylab

# 30MinNet
# authors: Robert Fratila, Gabriel Alacchi

# TRAINING HYPER PARAMS
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08


def get_convolution_ops(dimensions, input_var):
    print ("Creating Network...")
    network = lasagne.layers.InputLayer(shape=dimensions, input_var=input_var)

    print ('Input Layer:')
    print '	', lasagne.layers.get_output_shape(network)

    print ('Hidden Layer:')
    network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5, 5), pad ='same', nonlinearity=lasagne.nonlinearities.rectify)
    print '	', lasagne.layers.get_output_shape(network)
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
    print '	', lasagne.layers.get_output_shape(network)

    network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5, 5), pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    print '	', lasagne.layers.get_output_shape(network)
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
    print '	', lasagne.layers.get_output_shape(network)

    return network


def create_classification_head(network):
    # dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer

    # Add the classification softmax head
    network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

    print ('Output Layer:')
    print '	',lasagne.layers.get_output_shape(network)

    return network


def create_bounding_box_head(network):

    network = lasagne.layers.DenseLayer(network, num_units=4, nonlinearity = lasagne.nonlinearities.linear)

    print ('Output Layer:')
    print ' ', lasagne.layers.get_output_shape(network)

    return network


def create_trainer(network, input_var, y):
    print ("Creating Trainer...")
    # output of network
    out = lasagne.layers.get_output(network)
    # get all parameters from network
    params = lasagne.layers.get_all_params(network, trainable=True)
    # calculate a loss function which has to be a scalar
    cost = T.nnet.categorical_crossentropy(out, y).mean()
    # calculate updates using ADAM optimization gradient descent
    updates = lasagne.updates.adam(cost, params, learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON)
    # theano function to compare brain to their masks with ADAM optimization
    train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)

    return train_function


def save_model(network, save_location='', model_name='brain1'):

    network_name = '%s%s.npz' % (save_location, model_name)
    print ('Saving model as %s' % network_name)
    numpy.savez(network_name, *lasagne.layers.get_all_param_values(network))


def load_model(network, model='brain1.npz'):

    with numpy.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]  # gets all param values
        # lasagne.layers.set_all_param_values(network, param_values)   # sets all param values
    return network


def main():
    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    truth = T.dmatrix('truth')

    # Create conv net
    conv_net = get_convolution_ops(dimensions=(1, 1, 300, 600), input_var=input_var)

    # create classification head
    class_net = create_classification_head(conv_net)

    bbox_net = create_bounding_box_head(conv_net)

    # Create trainer
    trainer = create_trainer(network=class_net, input_var=input_var, y=truth)


if __name__ == "__main__":
    main()
