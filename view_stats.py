import numpy
import pylab #for graphing
import pickle
import sys

data = {}
model_name = "br1_stats.pickle"
if len(sys.argv) > 1:
	model_name = sys.argv[1]
with open('%s'%model_name) as in_file:
	data = pickle.load(in_file)
	
'''
for key in data.keys():
	data[key] = numpy.array(data[key],dtype='float32')
'''
pylab.plot(data['epoch'],data['error'], '-ro',label='Test Error')
pylab.plot(data['epoch'],data['accuracy'],'-go',label='Test Accuracy')
pylab.xlabel("Epoch")
pylab.ylabel("Error (%)")
pylab.ylim(0,max(data['error']) if max(data['error']) < 20000 else 20000)
pylab.title(model_name)
pylab.legend(loc='upper right')
#pylab.savefig('.png'%modelName)
pylab.show()#enter param False if running in iterative mode