import numpy
import theano
import theano.tensor as T
import lasagne
from os import path

from collections import OrderedDict
import pylab

import time

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
    network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
    print '	', lasagne.layers.get_output_shape(network)

    network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5, 5), pad='same', nonlinearity=lasagne.nonlinearities.rectify)
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

def create_validator(network, input_var, y):
    print ("Creating Validator...")
    #We will use this for validation intermi
    valid_prediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
    valid_loss = lasagne.objectives.categorical_crossentropy(valid_prediction,y).mean()   #check how much error in prediction
    valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

    validate_fn = theano.function([input_var, y], [valid_loss, valid_acc])	 #check for error and accuracy percentage
    return validate_fn

def save_model(network, save_location='', model_name='brain1'):

    network_name = '%s.npz' % path.join(save_location, model_name)
    print ('Saving model as %s' % network_name)
    numpy.savez(network_name, *lasagne.layers.get_all_param_values(network))


def load_model(network, model='brain1.npz'):

    with numpy.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]  # gets all param values
        # lasagne.layers.set_all_param_values(network, param_values)   # sets all param values
    return network

def get_data():
	#data = get_data('data/img/a_001.jpg','data/label/a_001.txt')
	d = numpy.array([numpy.zeros(shape=(700,900)) for i in xrange(10)],dtype='float32')
	y = numpy.array([[[0,1]]]*10,dtype='float32')
	return OrderedDict(input=d,truth=y)

def main():
    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    truth = T.dmatrix('truth')
    epochs_to_train = 10
    samples_per_epoch = 3
    train_time = 0.01 #in hours
    model_name='br1'    

    data = get_data()

    print ("%i samples found")%data['input'].shape[0]
    test_reserve = 0.2
    validation_reserve = 0.2
    training_reserve = 1-(test_reserve+validation_reserve)

    training_set = data['input'][:int(training_reserve*data['input'].shape[0])]
    validation_set = data['input'][int(training_reserve*data['input'].shape[0]):-int(validation_reserve*data['input'].shape[0])]
    test_set = data['input'][int(validation_reserve*data['input'].shape[0] + int(training_reserve*data['input'].shape[0])):]
    
    # Create conv net
    conv_net = get_convolution_ops(dimensions=(1, 1, 700, 900), input_var=input_var)

    # create classification head
    class_net = create_classification_head(conv_net)

    # create bounding box head
    #bbox_net = create_bounding_box_head(conv_net)

    # Create trainer
    trainer = create_trainer(network=class_net, input_var=input_var, y=truth)

    # Create validator
    validator = create_validator(class_net,input_var,truth)

    record = OrderedDict(epoch=[],error=[],accuracy=[])

    print ("Training for %s epoch(s) with %s samples per epoch"%(epochs_to_train,samples_per_epoch))
    #import pudb; pu.db
    epoch = 0
    start_time = time.time()
    time_elapsed = time.time() - start_time
    #for epoch in xrange(epochs):            #use for epoch training
    while epoch < epochs_to_train:     #use for time training
        epoch_time = time.time()
        print ("--> Epoch: %d | Epochs left: %d")%(epoch,epochs_to_train-epoch)

        for i in xrange(samples_per_epoch):
            choose_randomly = numpy.random.randint(training_set.shape[0])
            train_in = training_set[choose_randomly]
            #import pudb; pu.db
            train_in = train_in.reshape([1,1] + list(train_in.shape))
            trainer(train_in, data['truth'][choose_randomly])

        choose_randomly = numpy.random.randint(validation_set.shape[0])
        print ("Gathering data...")#%s"%validation_set[choose_randomly])
        train_in = validation_set[choose_randomly]
        train_in = train_in.reshape([1,1] + list(train_in.shape))
        error, accuracy = validator(train_in, data['truth'][choose_randomly])			     #pass modified data through network
        record['error'].append(error)
        record['accuracy'].append(accuracy)
        record['epoch'].append(epoch)
        time_elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_time
        print ("	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epoch_time))
        epoch+=1

    save_model(conv_net, 'data', 'conv_weights')

if __name__ == "__main__":
    main()
