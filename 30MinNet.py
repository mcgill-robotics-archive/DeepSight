import numpy
import theano
import theano.tensor as T
import lasagne

from collections import OrderedDict
import pylab

# 30MinNet
# authors: Robert Fratila, Gabriel Alacchi

# TRAINING HYPER PARAMS
LEARNING_RATE=0.001
BETA_1=0.9
BETA_2=0.999
EPSILON=1e-08

def createClassificationNetwork(dimensions, inputVar):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer

	print ("Creating Network...")
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=inputVar)

	print ('Input Layer:')
	print '	',lasagne.layers.get_output_shape(network)

	print ('Hidden Layer:')
	network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	print '	',lasagne.layers.get_output_shape(network)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)

	print ('Output Layer:')
	print '	',lasagne.layers.get_output_shape(network)

	return network


def createTrainer(network,inputVar,y):
	print ("Creating Trainer...")
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([inputVar, y], updates=updates) # omitted (, allow_input_downcast=True)

	return train_function

def main():
	inputVar = T.tensor4('input')#this will hold the image that gets inputted
	truth = T.dmatrix('truth')
	
	#create network
	network = createClassificationNetwork(dimensions=(1,1,300,600),inputVar=inputVar)

	#create trainer
	trainer = createTrainer(network = network, inputVar = inputVar, y = truth)


if __name__ == "__main__":
	main()
