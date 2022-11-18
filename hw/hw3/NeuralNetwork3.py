import neuralnetworks
import numpy as np
import sys

data = open(sys.argv[1], 'r').readlines()
data = [[float(val) for val in row.strip().split(',')] for row in data]
data = np.array(data)
labels = open(sys.argv[2], 'r').readlines()
labels = [[float(val) for val in row.strip().split(',')] for row in labels]
labels = np.array(labels)
test = open(sys.argv[3], 'r').readlines()
test = [[float(val) for val in row.strip().split(',')] for row in test]
test = np.array(test)
n_inputs = data.shape[1]
n_hiddens_per_layer = [8, 5]
n_outputs = len(np.unique(labels))
n_epochs = 10000
learning_rate = 0.01

NN = neuralnetworks.BinaryNeuralNetworkClassifier(data.shape[1], n_hiddens_per_layer, n_outputs)
NN.train(data, labels, n_epochs, learning_rate, method='adam', verbose=True)
result = NN.use(test)
result = result[0].astype(int)
np.savetxt('test_predictions.csv', result, fmt='%i')