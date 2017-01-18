import numpy
import theano
import theano.tensor as T
import lasagne
from os import path

from collections import OrderedDict
import pylab
import sys

import time

from DataSet import DataSet

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
    import Image
    #d = numpy.array([numpy.array(Image.open('data/img/a_00%d.jpg'%i)) for i in xrange(1,10)],dtype='float32')
    d = numpy.array([numpy.zeros(shape=(3, 972, 723)) for i in xrange(10)],dtype='float32')
    y = numpy.array([[0,1]]*10,dtype='float32')
    return OrderedDict(input=d,truth=y)

def main():
    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    truth = T.dmatrix('truth')
    epochs_to_train = 3
    train_time = 0.01 #in hours
    model_name='br1' 

    batch_size = 2

    data = get_data()

    num_samples = data['input'].shape[0]

    print ("%i samples found" % num_samples)
    
    test_reserve = 0.2
    validation_reserve = 0.2
    training_reserve = 1-(test_reserve+validation_reserve)

    training_data_set = DataSet(['a_{0:03}'.format(i + 1) for i in xrange(6)], data_dir='./data', batch_size=batch_size)
    validation_data_set = DataSet(['a_{0:03}'.format(i + 1) for i in xrange(6, 8)], data_dir='./data', batch_size=2)
    testing_data_set = DataSet(['a_{0:03}'.format(i + 1) for i in xrange(8, 10)], data_dir='./data', batch_size=2)

    num_train_steps = training_data_set.get_epoch_steps()

    # import pudb; pu.db
    # Create conv net
    conv_net = get_convolution_ops(dimensions=(None, data['input'].shape[1], data['input'].shape[2], data['input'].shape[3]), input_var=input_var)

    # create classification head
    class_net = create_classification_head(conv_net)

    # create bounding box head
    # bbox_net = create_bounding_box_head(conv_net)

    # Create trainer
    trainer = create_trainer(network=class_net, input_var=input_var, y=truth)

    # Create validator
    validator = create_validator(class_net,input_var,truth)

    record = OrderedDict(epoch=[],error=[],accuracy=[])

    print ("Training for %s epoch(s) with %s steps per epoch"%(epochs_to_train,num_train_steps))
    #import pudb; pu.db
    epoch = 0
    start_time = time.time()
    time_elapsed = time.time() - start_time
    #for epoch in xrange(epochs):            #use for epoch training
    while epoch < epochs_to_train:     #use for time training
        epoch_time = time.time()
        print ("--> Epoch: %d | Epochs left: %d"%(epoch,epochs_to_train-epoch))

        for i in xrange(num_train_steps):
            # Get next batch

            train_in, truth_in = training_data_set.next_batch()
            trainer(train_in, truth_in)
            percentage = float(i+1) / float(num_train_steps) * 100
            sys.stdout.flush()
            sys.stdout.write ("\r %d training steps complete: %.2f%% done epoch" % (i + 1, percentage))

        # Get error, accuracy on the test set at the end of every epoch
        print "\nGetting test accuracy..."

        # TODO: If the test / validation sets are too large we should load them in batches rather than the entire set
        test_set, test_truth = testing_data_set.load_all()

        error, accuracy = validator(test_set, test_truth)
        record['error'].append(error)
        record['accuracy'].append(accuracy)
        record['epoch'].append(epoch)
        time_elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_time
        print ("\n	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epoch_time))

        epoch += 1

    print "Validating..."

    # Finally validate the final error and accuracy with the validation set
    # We should validate on the entire set, so use load_all
    validation_set, validation_truth = validation_data_set.load_all()
    error, accuracy = validator(validation_set, validation_truth)
    print ("\n\nFinal Results after %d epochs of training and %.2fs elapsed" % (epochs_to_train, time_elapsed))
    print ("    error: %s and accuracy: %s" % (error, accuracy))

    save_model(conv_net, 'data', 'conv_weights')
    save_model(class_net, 'data', 'classifier_net')

if __name__ == "__main__":
    main()
