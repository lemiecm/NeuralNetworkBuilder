from csv import reader
from math import fabs
import numpy as np
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
from csv import reader
import random
from nn import *
from not_mine import * 


# def spliting(dataset, num_of_input, num_of_classes):
#     random.shuffle(dataset)
#     size_data = len(dataset)
#     train_len = int(size_data*0.8)
#     train = dataset[:train_len]
#     test = dataset[train_len:]

#     train_y = [item[0] for item in train]
#     train_y = [one_hot_encode(y,3) for y in train_y]

#     train_x = [item[1:] for item in train]
#     train_x = [np.reshape(x, (1, 13)) for x in train_x]
#     print(np.array(train_y).shape)
    
#     test_x = [item[1:] for item in test]
#     test_x = [np.reshape(x, (13, 1)) for x in test_x]
#     test_y =[item[0] for item in test]
#     # test_y = [one_hot_encode(y,3) for y in test_y]

#     train_zipped = zip(train_x, train_y)
#     test_zipped = zip(test_x, test_y)

#     return train_zipped, test_zipped

def spliting(dataset, num_of_input = 13, num_of_classes = 3, label_col = 0):
    random.shuffle(dataset)
    size_data = len(dataset)
    train_len = int(size_data*0.8)
    train = dataset[:train_len]
    test = dataset[train_len:]

    train_y = [item[label_col] for item in train]
    train_y = [one_hot_encode(y,num_of_input) for y in train_y]

    train_x = [item[:11] for item in train]
    train_x = [np.reshape(x, (1,num_of_classes)) for x in train_x]
    print(np.array(train_y).shape)
    
    test_x = [item[:11] for item in test]
    test_x = [np.reshape(x, (1, num_of_input)) for x in test_x]

    test_y = [item[11] for item in test]
    test_y = [one_hot_encode(y,num_of_classes) for y in test_y]
    # test_y = [one_hot_encode(y,3) for y in test_y]

    train_zipped = zip(train_x, train_y)
    test_zipped = zip(test_x, test_y)

    return train_zipped, test_zipped


def one_hot_encode(y, num_of_classes):
    encoded = np.zeros((num_of_classes, 1))
    encoded[y-1] = 1.0
    return encoded

def load_csv(filename, columns_to_int, columns_to_float, is_header=False, is_csv =False):
    dataset = list()
    with open(filename, 'r') as file:
        if is_csv:
            csv_reader = reader(file,delimiter =";" )
        else:
            csv_reader = reader(file,delimiter ="," )
        if is_header:
            next(csv_reader, None)
        for row in csv_reader:
            if not row:
                continue
            for column in columns_to_float:
                row[column] = float(row[column].strip())
            for column in columns_to_int:
                row[column] = int(row[column].strip())
            dataset.append(row)
    return dataset

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# def parse_txt(fname, num_features=13, num_targets=1):
    
#     '''
#         Read data from a text file and generate arrays 
#         ready to be fed into the network as inputs.

#         Each line in the text file is separated by a
#         newline, and represents a data point.
#         Features in a line are separated by blank space 
#         and the last data point is the target.

#     '''

#     X = np.empty((178, num_features), dtype=float)
#     Y = np.empty(178, dtype=int)

#     with open(fname) as f:
#         for index, line in enumerate(f):
#             line = line.rstrip('\n')
#             data = line.split(',')


#             X[index, :] = np.asarray(data[1:])
#             Y[index] = np.asarray(data[0])

#     return X, Y

# def train_val_split(X, Y, train_percent=0.8):

#     '''
#         Function takes in the training data as input and returns
#         a training validation split based on a given percentage.
#     '''

#     num_points = X.shape[0]

#     train_size = int(num_points * 100 * train_percent // 100)

#     inds = np.arange(num_points)
#     np.random.shuffle(inds)

#     train_inds = inds[:train_size]
#     val_inds = inds[train_size: ]

#     train_X = X[train_inds, :]
#     val_X = X[val_inds, :]

#     train_Y = Y[train_inds]
#     val_Y = Y[val_inds]

#     return train_X, train_Y, val_X, val_Y

# X, Y = parse_txt('./data/wine.data')
# train_X, train_Y, val_X, val_Y = train_val_split(X, Y)
# net = Network([13,5,3])
# net.SGD(zip(train_X,train_Y), 5, 32, 0.3)
# reading data or csv files
data = load_csv("./data/winequality-white.csv",[11],[*range(0,11,1)],is_header=True, is_csv =True)

train, test = spliting(data, num_of_input=11, num_of_classes=11,label_col=11)
# print(list(train))
# print([list(element) for element in zip(*train)][1])
# print(list(train)[0])
# random_vectors = lambda dim, cnt: [np.random.rand(dim, 1) for i in range(cnt)]
# random_batch= list(zip(random_vectors(3, 64) , random_vectors(2, 64)))
# print(len(random_batch))
nn = NeuralNetwork([11,20,16,11])
nn.train(list(train),250,8,0.0001)


# data = load_csv("./data/wine.data",[0,13],[*range(1,13,1)])

# train, test = spliting(data, num_of_input=13, num_of_classes=3)
# # print(list(train))
# # print([list(element) for element in zip(*train)][1])
# # print(list(train)[0])
# # random_vectors = lambda dim, cnt: [np.random.rand(dim, 1) for i in range(cnt)]
# # random_batch= list(zip(random_vectors(3, 64) , random_vectors(2, 64)))
# # print(len(random_batch))
# nn = NeuralNetwork([13,20,10,3])
# nn.train(list(train),200,4,0.0001)

# net = Network([13,5,5,3])
# net.SGD(train, 5, 32, 0.3,test)

# my_net = Network([3, 2 ,2])
# print("Initial Weights:")
# print(my_net.Wₙ[0])
# #the following generates a list of cnt vectors of length dim.
# random_vectors = lambda dim, cnt: [np.random.rand(dim, 1) for i in range(cnt)]

# print(random_vectors(3, 64)[0].shape)
# random_batch= list(zip(random_vectors(3, 64) , random_vectors(2, 64)))
#print(random_batch.shape)
# my_net.gradient_descent(list(train), 3.0)
# print("Optimized Weights:")
# print(my_net.Wₙ[0])