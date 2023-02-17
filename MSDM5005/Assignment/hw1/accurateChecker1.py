from mnistClassification import Linear, Sigmoid, Softmax, CrossEntropy, Mean, load_MNIST, match_ratio, net_forward
import numpy as np
import json


train_data, train_label, test_data, test_label = load_MNIST()

with open('npMnistStructure.json', 'r') as f:
    config = json.load(f)

net = []
with open('npMnistParameters.npy', 'rb') as f:
    for idx, term in enumerate(config['struct']):
        if term == 'linear':
            net.append(Linear(config['num_parametes'][idx][0], config['num_parametes'][idx][1]))
            parameters = [np.load(f), np.load(f)]
            net[-1].parameters = parameters
        elif term == 'sigmoid':
            net.append(Sigmoid())
        elif term == 'softmax':
            net.append(Softmax())
        elif term == 'cross_entropy':
            net.append(CrossEntropy())
        elif term == 'mean':
            net.append(Mean())
        else:
            raise Exception("The loaded node name not recognized!")

result_test, loss_test = net_forward(net, test_data.reshape(test_data.shape[0], -1), test_label)
print('MNIST test correct rate = %.3f' % (match_ratio(result_test, test_label)))
