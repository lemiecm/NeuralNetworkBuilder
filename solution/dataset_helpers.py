

import matplotlib.pylab as plt

import pandas as pd
import numpy as np
from csv import reader
import random
from nn import *


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

def one_hot_part_2(y, num_of_classes):
    encoded = np.zeros((num_of_classes, 1))
    encoded[y-1-3] = 1.0
    return encoded

def one_hot_part_3(y, num_of_classes):
    encoded = np.zeros((num_of_classes, 1))
    res =  int(str({
        'SEKER':0,
        'BARBUNYA':1, 
        'BOMBAY':2, 
        'CALI':3, 
        'DERMASON':4, 
        'HOROZ':5,
        'SIRA':6
    }.get(y)))
    encoded[res] = 1.0
    return encoded

def one_hot_part_4(y, num_of_classes):
    encoded = np.zeros((num_of_classes, 1))
    res =  int(str({
        'Iris-setosa':0,
        'Iris-versicolor':1, 
        'Iris-virginica':2, 
    }.get(y)))
    encoded[res] = 1.0
    return encoded
    
    
def spliting(dataset, num_of_input = 13, num_of_classes = 3, label_col = 0, is_first_label = True,is_string_columns=False):
    random.shuffle(dataset)
    size_data = len(dataset)
    
    # Divide data in ration 0.2 0.8 
    train_len = int(size_data*0.8)
    train = dataset[:train_len]
    test = dataset[train_len:]

    if(is_first_label):
        train_y = [item[label_col] for item in train]
        train_y = [one_hot_encode(y,num_of_classes) for y in train_y]

        train_x = [item[label_col+1:] for item in train]
        train_x = [np.reshape(x, (1, num_of_input)) for x in train_x]

        test_y = [item[label_col] for item in test]
        test_y = [one_hot_encode(y,num_of_classes) for y in test_y]

        test_x = [item[label_col+1:] for item in test]
        test_x = [np.reshape(x, (1, num_of_input)) for x in test_x]

    elif not is_string_columns:
        train_y = [item[label_col] for item in train]
        train_y = [one_hot_encode(y,num_of_classes) for y in train_y]

        train_x = [item[:label_col] for item in train]
        train_x = [np.reshape(x, (1, num_of_input)) for x in train_x]
        
        test_y = [item[label_col] for item in test]
        test_y = [one_hot_encode(y,num_of_classes) for y in test_y]

        test_x = [item[:label_col] for item in test]
        test_x = [np.reshape(x, (1, num_of_input)) for x in test_x]

    else:
        train_y = [item[label_col] for item in train]
        train_y = [one_hot_part_4(y,num_of_classes) for y in train_y]

        train_x = [item[:label_col] for item in train]
        train_x = [np.reshape(x, (1, num_of_input)) for x in train_x]
        
        test_y = [item[label_col] for item in test]
        test_y = [one_hot_part_4(y,num_of_classes) for y in test_y]

        test_x = [item[:label_col] for item in test]
        test_x = [np.reshape(x, (1, num_of_input)) for x in test_x]
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

def test1(epochs=100,batch_size=16,lr=0.001,activation_function='tanh', momentum =0.9):
    data = load_csv("./data/winequality-red.csv",[11],[*range(0,11,1)],is_header=True, is_csv =True)

    train, test = spliting(data, num_of_input=11, num_of_classes=11,label_col=11, is_first_label =False)
    nn = NeuralNetwork([11,20,16,11],activation_function, momentum)
    nn.train(list(train),epochs,batch_size,lr)
    nn.test(list(test),batch_size)
    
def test2():
    data = load_csv("./data/wine.data",[0,13],[*range(1,13,1)],is_header=False, is_csv =False)

    train, test = spliting(data, num_of_input=13, num_of_classes=3,label_col=0, is_first_label =True)
    nn = NeuralNetwork([13,20,10,3],activation_function='tanh')
    nn.train(list(train),200,4,0.001)

def test3():
    data = load_csv("./data/Dry_Bean_Dataset.csv",[0,6],[1,2,3,4,5,7,8,9,10,11,12,13,14,15],is_header=True, is_csv =True)

    train, test = spliting(data, num_of_input=16, num_of_classes=7,label_col=16, is_first_label =False, is_string_columns=True)
    nn = NeuralNetwork([16,20,10,7])
    nn.train(list(train),100,64,0.001)

def test4():
   
    data = load_csv("./data/wheat-seeds.csv",[7],[*range(0,7,1)],is_header=False, is_csv =False)

    train, test = spliting(data, num_of_input=7, num_of_classes=3,label_col=7, is_first_label =False)
    nn = NeuralNetwork([7,20,10,3])
    nn.train(list(train),100,8,0.002)
    nn.test(list(test))

def test5(epochs=100,batch_size=4,lr=0.01,activation_function='tanh', momentum=0.9):
    data = load_csv("./data/iris.data",[],[*range(0,4,1)],is_header=False, is_csv =False)

    train, test = spliting(data, num_of_input=4, num_of_classes=3,label_col=4,is_first_label =False, is_string_columns=True)
    nn = NeuralNetwork([4,8,6,3],activation_function, momentum)
    nn.train(list(train),epochs,batch_size,lr)
    nn.test(list(test),batch_size)
   
    
    

def test2():
    data = load_csv("./data/wine.data",[0,13],[*range(1,13,1)],is_header=False, is_csv =False)

    train, test = spliting(data, num_of_input=13, num_of_classes=3,label_col=0, is_first_label =True)
    nn = NeuralNetwork([13,20,10,3],activation_function='tanh')
    nn.train(list(train),200,4,0.0085)

def run_from_app(data_num,layers,args):
    path = str({
        1:'./data/winequality-red.csv',
        2:'./data/iris.data' 
    }.get(data_num))

    if data_num == 1:
        data = load_csv('./data/winequality-red.csv',[11],[*range(0,11,1)],is_header=True, is_csv =True)
        train, test = spliting(data, num_of_input=11, num_of_classes=11,label_col=11, is_first_label =False)
    else:
        data = load_csv('./data/iris.data' ,[],[*range(0,4,1)],is_header=False, is_csv =False)
        train, test = spliting(data, num_of_input=4, num_of_classes=3,label_col=4,is_first_label =False, is_string_columns=True)

    nn = NeuralNetwork(layers,args["activation_function"],args["momentum"])
    nn.train(list(train),epochs=args['epochs'],batch_size=args['batch_size'], learning_rate = args["learning_rate"])
    nn.test(list(test),batch_size=args['batch_size'])

def test5(epochs=100,batch_size=4,lr=0.01,activation_function='tanh', momentum=0.9):
    data = load_csv("./data/iris.data",[],[*range(0,4,1)],is_header=False, is_csv =False)

    train, test = spliting(data, num_of_input=4, num_of_classes=3,label_col=4,is_first_label =False, is_string_columns=True)
    nn = NeuralNetwork([4,8,6,3],activation_function, momentum)
    nn.train(list(train),epochs,batch_size,lr)
    nn.test(list(test),batch_size)

def change_of_learing_rate():
    for lr in [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        test1(momentum=lr)
        
def plot_some_graphs(arg):
    plt.figure()
    # length of loss = num of epochs
    plt.plot(arg["lr"],arg["losses"])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.title(f"Change of learning rate for Wine")
    plt.show()

