
import ThirtyMinNet
import theano
import theano.tensor as T
import sys
import argparse
from os import path, listdir
from load_data import load_image
import numpy as np

from DataSet import DataSet


def recreate_classifier(input_var, model_file, batch_size):
    # Create conv net
    conv_net = ThirtyMinNet.get_convolution_ops(dimensions=(batch_size, 3, 210, 280), input_var=input_var)

    # create classification head
    class_net = ThirtyMinNet.create_classification_head(conv_net)

    model = ThirtyMinNet.load_model(class_net, model_file)

    return model


def evaluate(args):
    print "Evaluating model %s" % args.model_file

    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    truth = T.dmatrix('truth')

    batch_size = 10

    model = recreate_classifier(input_var, model_file=args.model_file, batch_size=batch_size)

    validator = ThirtyMinNet.create_validator(model, input_var, truth)

    image_path = path.join(args.data_dir, 'img')
    image_names = [path.basename(img).replace('.jpg', '') for img in listdir(image_path) if img.endswith('.jpg')]

    all_data = DataSet(image_names, data_dir=args.data_dir, net_type='Custom', batch_size=batch_size)

    print "Evaluating the entire DataSet (this could take some time)"

    num_steps = all_data.get_epoch_steps()

    error, accuracy = 0.0, 0.0
    for step in xrange(num_steps):
        sys.stdout.flush()
        sys.stdout.write('\r step %d out of %d' % (step + 1, num_steps))

        data_in, data_truth = all_data.next_batch()

        step_err, step_acc = validator(data_in, data_truth)

        error += step_err
        accuracy += step_acc

    error /= num_steps
    accuracy /= num_steps

    print "\nFinal results: --- Error: %f --- Accuracy: %f" % (error, accuracy)


def classify(args):
    image_names = args.image_names.split(',')
    num_images = len(image_names)
    print "Classifying %d images" % len(image_names)

    image_dir = path.join(args.data_dir, 'img')

    input_var = T.tensor4('input')  # this will hold the image that gets inputted
    model = recreate_classifier(input_var, model_file=args.model_file, batch_size=1)

    print "Creating theano function from model"
    classifier = theano.function(inputs=[input_var], outputs=[model])

    for i in xrange(num_images):
        image_path = path.join(image_dir, image_names[i] + '.jpg')
        image = load_image(image_path, net_type='Custom')

        result = classifier(image)
        if np.argmax(result[0], axis=1) == 1:
            print "%s: No buoy" % image_names[i]
        else:
            print "%s: Has a buoy" % image_names[i]


def main(argv):

    parser = argparse.ArgumentParser(prog='evaluate.py', description='Evaluation script for thirty min net')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create parser for the eval command
    eval_parser = subparsers.add_parser('eval', help='eval help')
    eval_parser.add_argument('-d', '--data-dir', dest='data_dir', help='Root directory of the data', required=True)
    eval_parser.add_argument('-m', '--model-file', dest='model_file', help='.npz archive containing the model weights', required=True)
    eval_parser.set_defaults(func=evaluate)

    classify_parser = subparsers.add_parser('classify', help='classify help')
    classify_parser.add_argument('-d', '--data-dir', dest='data_dir', help='Root directory of that data', required=True)
    classify_parser.add_argument('-m', '--model-file', dest='model_file', help='.npz archive containing the classifier model weights', required=True)
    classify_parser.add_argument('-n', '--image-names', dest='image_names',
                                 help='Comma delimited list of image names to classify. Example: a_001,b_022,a_011,c_033', required=True)
    classify_parser.set_defaults(func=classify)

    args = parser.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main(sys.argv[1:])
