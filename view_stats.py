import numpy
import pylab #for graphing
import pickle
import sys
import argparse

def main(argv):
    argparser = argparse.ArgumentParser(description='Visualize the training stats')
    argparser.add_argument('-f', '--training-file',
        dest='training_file',
        help='.pickle file containing the training results',
        required=True)

    args = argparser.parse_args(argv)

    data = {}
    model_name = args.training_file

    with open(model_name, mode='r') as pickle_file:
        data = pickle.load(pickle_file)

    '''
    for key in data.keys():
        data[key] = numpy.array(data[key],dtype='float32')
    '''
    pylab.plot(data['epoch'],data['error'], '-ro',label='Test Error')
    pylab.plot(data['epoch'],data['accuracy'],'-go',label='Test Accuracy')
    pylab.xlabel("Epoch")
    pylab.ylabel("Error (%)")
    pylab.ylim(0, max(data['error']) if max(data['error']) < 20000 else 20000)
    pylab.title(model_name)
    pylab.legend(loc='upper right')
    #pylab.savefig('.png'%modelName)
    pylab.show()#enter param False if running in iterative mode

if __name__ == "__main__":
    main(sys.argv[1:])

