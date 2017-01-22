#!/usr/bin/python

import numpy as np
import theano
import theano.tensor as T
import lasagne
from os import path

from collections import OrderedDict
import sys

import time

from DataSet import create_data_sets
import argparse

# 30MinNet
# authors: Robert Fratila, Gabriel Alacchi

# TRAINING HYPER PARAMS DEFAULTS
LEARNING_RATE = 1e-5
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
BATCH_SIZE = 10
EPOCHS = 20


def get_convolution_ops(dimensions, input_var):
    print ("Creating Network...")
    network = lasagne.layers.InputLayer(shape=dimensions, input_var=input_var)

    print ('Input Layer:')
    print ' ', lasagne.layers.get_output_shape(network)

    print ('Hidden Layer:')
    network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5, 5), pad ='same', nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print ' ', lasagne.layers.get_output_shape(network)

    network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5, 5), pad='same', nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    print ' ', lasagne.layers.get_output_shape(network)

    return network


def create_classification_head(network):

    network = lasagne.layers.DenseLayer(network, num_units=1024, nonlinearity=lasagne.nonlinearities.rectify)

    # Add the classification softmax head
    network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    print ('Output Layer:')
    print ' ', lasagne.layers.get_output_shape(network)

    return network


def create_bounding_box_head(network):

    network = lasagne.layers.DenseLayer(network, num_units=4, nonlinearity = lasagne.nonlinearities.linear)

    print ('Output Layer:')
    print ' ', lasagne.layers.get_output_shape(network)

    return network


def create_trainer(network, input_var, y, learning_rate=LEARNING_RATE, beta1=BETA_1, beta2=BETA_2, epsilon=EPSILON):
    print ("Creating Trainer...")
    # output of network
    out = lasagne.layers.get_output(network)
    # get all parameters from network
    params = lasagne.layers.get_all_params(network, trainable=True)
    # calculate a loss function which has to be a scalar
    cost = T.nnet.categorical_crossentropy(out, y).mean()
    # calculate updates using ADAM optimization gradient descent
    updates = lasagne.updates.adam(cost, params, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    # theano function to compare brain to their masks with ADAM optimization
    train_function = theano.function([input_var, y], updates=updates)   # omitted (, allow_input_downcast=True)

    return train_function


def create_validator(network, input_var, y):
    print ("Creating Validator...")
    # We will use this for validation
    valid_prediction = lasagne.layers.get_output(network, deterministic=True)           # create prediction
    valid_loss = lasagne.objectives.categorical_crossentropy(valid_prediction, y).mean()   # check how much error there is in prediction
    valid_acc = T.mean(T.eq(T.argmax(valid_prediction, axis=1), T.argmax(y, axis=1)), dtype=theano.config.floatX)    # check the accuracy of the prediction

    validate_fn = theano.function([input_var, y], [valid_loss, valid_acc])   # check for error and accuracy percentage
    return validate_fn


def save_model(network, save_location='', model_name='brain1'):

    network_name = '%s.npz' % path.join(save_location, model_name)
    print ('Saving model as %s' % network_name)
    np.savez(network_name, *lasagne.layers.get_all_param_values(network))


def load_model(network, model='brain1.npz'):

    with np.load(model) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]  # gets all param values
        lasagne.layers.set_all_param_values(network, param_values)   # sets all param values
    return network


def main(argv):

    # Arg parse options, all of these have defaults and are merely here for convenience when training
    parser = argparse.ArgumentParser(prog="Thirty Min Net training script.")
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=BATCH_SIZE, type=int, help='Batch size to train with, defaults to 30')
    parser.add_argument('-e', '--epochs', dest='epochs_to_train', default=EPOCHS, type=int, help='The number of training epochs')
    parser.add_argument('-d', '--data-dir', dest='data_dir', default='data', help='Root directory of the data for training')
    parser.add_argument('--reserve', dest='reserve', default='0.7,0.1,0.2',
                        help='The proportion of data to allocate to the training, testing, validation data sets respectively. Default value is 0.7,0.1,0.2')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', default=LEARNING_RATE, type=float, help='Learning rate to train with')
    parser.add_argument('--adam-opts', dest='adam_opts', default='0.9,0.999,1e-8', help='Adam optimizer options as a comma delimited list in the order beta1,beta2,epsilon. Default value is 0.9,0.999,1e-8')
    parser.add_argument('-n', '--model-name', dest='model_name', default='blah', help='The name of the model')

    args = parser.parse_args(argv)

    adam_opts = map(lambda opt: float(opt), args.adam_opts.split(','))
    reserve = map(lambda opt: float(opt), args.reserve.split(','))

    if len(adam_opts) != 3:
        print "Invalid value for option --adam-opts %s" % args.adam_opts
        sys.exit(-1)
    elif len(reserve) != 3:
        print "Invalid value for option --reserve %s" % args.adam_opts
        sys.exit(-1)

    print "Thirty Min Net Training"
    print "Model Name: %s" % args.model_name
    print "Batch Size: %d" % args.batch_size
    print "Learning Rate: %f" % args.learning_rate
    print "Adam Optimizer Options: "
    print "    beta1:      %.3f" % adam_opts[0]
    print "    beta2:      %.3f" % adam_opts[1]
    print "    epsilon:    %.3f" % adam_opts[2]
    print "Data Set Reserves: "
    print "    training:   %f" % reserve[0]
    print "    testing:    %f" % reserve[1]
    print "    validation: %f" % reserve[2]
    print "------------------------"

    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    truth = T.dmatrix('truth')

    epochs_to_train = args.epochs_to_train
    model_name = args.model_name

    batch_size = args.batch_size
    
    training, testing, validation = create_data_sets(data_dir=args.data_dir, net_type="Custom",
                                                     training_reserve=reserve[0], testing_reserve=reserve[1], validation_reserve=reserve[2])

    training.set_batch_size(batch_size)
    testing.set_batch_size(batch_size)
    validation.set_batch_size(batch_size)
    
    num_train_steps = training.get_epoch_steps()

    # Create conv net
    conv_net = get_convolution_ops(dimensions=(batch_size, 3, 210, 280), input_var=input_var)

    # create classification head
    class_net = create_classification_head(conv_net)

    # create bounding box head
    bbox_net = create_bounding_box_head(conv_net)

    # Create trainer
    class_trainer = create_trainer(network=class_net, input_var=input_var, y=truth,
                             learning_rate=args.learning_rate, beta1=adam_opts[0], beta2=adam_opts[1], epsilon=adam_opts[2])
    bbox_trainer = create_trainer(network=bbox_net, input_var=input_var, y=truth,
                             learning_rate=args.learning_rate, beta1=adam_opts[0], beta2=adam_opts[1], epsilon=adam_opts[2])
    # Create validator
    class_validator = create_validator(class_net,input_var,truth)

    bbox_validator = create_validator(bbox_net,input_var,truth)

    record = OrderedDict(epoch=[],error=[],accuracy=[])

    print ("Training for %s epoch(s) with %s steps per epoch"%(epochs_to_train,num_train_steps))

    # TODO: If the test / validation sets are too large we should load them in batches rather than the entire set
    # test_set, test_truth = testing.load_all()

    start_time = time.time()
    time_elapsed = time.time() - start_time

    for epoch in xrange(epochs_to_train):
        epoch_time = time.time()
        print ("--> Epoch: %d | Epochs left: %d"%(epoch,epochs_to_train-epoch))

        for i in xrange(num_train_steps):
            # Get next batch

            train_in, truth_in_class, truth_in_bbox = training.next_batch()

            #class_trainer(train_in, truth_in_class)
            bbox_trainer(train_in, truth_in_bbox)

            percentage = float(i+1) / float(num_train_steps) * 100
            sys.stdout.flush()
            sys.stdout.write("\r %d training steps complete: %.2f%% done epoch" % (i + 1, percentage))

            # On the last step
            if i == num_train_steps - 1:
                #test_err, test_acc = class_validator(train_in, truth_in_class)
                test_err, test_acc = bbox_validator(train_in, truth_in_bbox)
                print "\nTraining Batch Error"
                print "error: %s and accuracy: %s" % (test_err, test_acc)

        # Get error, accuracy on the test set at the end of every epoch
        print "\nGetting test accuracy on the entire testing set..."

        testing_steps = testing.get_epoch_steps()

        error, accuracy = 0.0, 0.0

        # Computes running average over the entire testing set
        for step in xrange(testing_steps):
            testing_set, testing_truth_class, testing_truth_bbox = testing.next_batch()

            #step_error, step_accuracy = class_validator(testing_set, testing_truth_class)

            step_error, step_accuracy = bbox_validator(testing_set, testing_truth_bbox)

            error += step_error
            accuracy += step_accuracy

        error /= testing_steps
        accuracy /= testing_steps

        record['error'].append(error)
        record['accuracy'].append(accuracy)
        record['epoch'].append(epoch)
        time_elapsed = time.time() - start_time
        epoch_time = time.time() - epoch_time
        print "\n  error: %s and accuracy: %s in %.2fs\n" % (error, accuracy, epoch_time)

        training.shuffle()

    print "Validating..."

    # Finally validate the final error and accuracy with the validation set
    # We should validate on the entire set, so use load_all

    validation_steps = validation.get_epoch_steps()

    error, accuracy = 0.0, 0.0

    # Computes running average over the entire validation set
    for step in xrange(validation_steps):
        validation_set, validation_truth_class, validation_truth_bbox = validation.next_batch()
        #step_error, step_accuracy = class_validator(validation_set, validation_truth_class)

        step_error, step_accuracy = bbox_validator(validation_set, validation_truth_bbox)

        error += step_error
        accuracy += step_accuracy

    error /= validation_steps
    accuracy /= validation_steps

    print ("\n\nFinal Results after %d epochs of training and %.2fs elapsed" % (epochs_to_train, time_elapsed))
    print ("    error: %s and accuracy: %s" % (error, accuracy))

    save_model(conv_net, 'data', 'conv_weights_%s'%model_name)
    save_model(class_net, 'data', 'classifier_net_%s'%model_name)

    # save metrics to pickle file to be opened later and displayed
    import pickle
    with open('data/%s_stats.pickle' % model_name, 'w') as output:

        # import pudb; pu.db
        pickle.dump(record, output)

if __name__ == "__main__":
    main(sys.argv[1:])
