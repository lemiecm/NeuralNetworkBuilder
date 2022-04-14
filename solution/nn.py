
import random
from re import S
from termios import TAB1
import numpy as np
import matplotlib.pylab as plt

# Activation Functions
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_derivative(activation):
    return sigmoid(activation) * (1 - sigmoid(activation))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(activation):
    return 1 - np.power(tanh(activation),2)

def softmax(z):
	expZ = np.exp(z - np.max(z))
	return expZ / expZ.sum(axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self,layers_sizes=[4,5,1], activation_function='sigmoid', momentum=0.9):
        np.random.seed(0)
        
        self.layers = {}
        self.activation_function = activation_function
        self.nn_size = len(layers_sizes)
        self.losses = []
        self.momentum = momentum
        
        # This way we get list of np arrays that will be helpful in math equations
        # for instance dot product from numpy
       
        self.weights = [np.random.randn(in_neurons,out_neurons)  * 0.1 for in_neurons,out_neurons in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.biases = [np.random.randn(in_neurons, 1) * 0.1 for in_neurons in layers_sizes[1:]] 
       
       # Nedded for momemntum
        self.prev_dws = [np.zeros(w.shape) for w in self.weights]
        self.prev_dbs = [np.zeros(b.shape) for b in self.biases]

    #    # in literature also known as velocity will be needed later on 
    #     self.previous_d_w = [np.zeros(w.shape) for w in self.weights]

    def __cross_entropy_loss(self,predicted, labeled):
        # 1e-7 - added to avoid having log(0)  
        target = np.argmax(labeled)
        y_pred = predicted[target]
        return -np.log(y_pred)

    def __accuracy(self, predicted, labeled):
        res = 1 if np.argmax(predicted) == np.argmax(labeled) else 0
        return res

    # Choose activation function
    def __activate(self,z):
        
        if self.activation_function == 'sigmoid':
            output = sigmoid(z)
        elif self.activation_function == 'tanh':
            output = tanh(z)
        else:
            output = sigmoid(z)

        return output

    def __deriv_activate(self,a):
        
        if self.activation_function == 'sigmoid':
            output = sigmoid_derivative(a)
        elif self.activation_function == 'tanh':
            output = tanh_derivative(a)
        else:
            output = sigmoid_derivative(a)

        return output

    # Forward pass through whole network
    # input - X in mathematical notation
    def __feed_forward_pass(self,input):
        cache = {}
        cache["weights"] = []
        cache["zs"] = []
        cache["outputs"] = []
        # needed for input layer
        output = input.T
        cache["outputs"].append(output)

        for i in range(self.nn_size-2):
            # keep weights
            cache["weights"].append(self.weights[i])

            z = np.dot(self.weights[i].T, output) + self.biases[i]
            # keep sums 
            cache["zs"].append(z)

            # run activation for every neron in every layer
            output = self.__activate(z)
            # keep activation function outputs 
            cache["outputs"].append(output)
           
        out_index = self.nn_size-2
        # keep weights
        cache["weights"].append(self.weights[out_index])
        z = np.dot(self.weights[out_index].T, output) + self.biases[out_index]
        # keep sums 
        cache["zs"].append(z)
        # run activation for every neron in every layer
        output = softmax(z)
        # keep activation function outputs 
        cache["outputs"].append(output)
        
        return output, cache

    # Backward pass through whole network
    # for now for sigmoid
    def __backward_pass(self, y,cache):
        cache["dws"] = [np.zeros(w.shape) for w in self.weights]
        cache["dbs"] = [np.zeros(b.shape) for b in self.biases]

        dL_dz = (cache["outputs"][-1] - y)
        dL_dw = np.dot(cache["outputs"][-2],dL_dz.T)
        dL_db = dL_dz

        cache["dws"][-1]= dL_dw
        cache["dbs"][-1] = dL_db

        # for hidden layers
        for i in range(2, self.nn_size):
           
            deriv = self.__deriv_activate(cache["zs"][-i])
            dL_dz = np.dot(self.weights[-i+1],dL_dz)*deriv
            dL_dw = np.dot(cache["outputs"][-i-1], dL_dz.T)
            dL_db = dL_dz
            cache["dws"][-i]= dL_dw
            cache["dbs"][-i] = dL_db

           
        return cache

    # main part of the NN 
    # input - x, input vector
    # y - label
    def __backprop(self, input, y):
        
        # forward pass
        output, cache = self.__feed_forward_pass(input)

        # backward pass
        cache = self.__backward_pass(y,cache)

        return output,cache

    def __mini_batch_train(self, mini_batch, learning_rate):
        # 'out of the loop' sum of weights and biases
        d_B = [np.zeros(b.shape) for b in self.biases]
        d_W = [np.zeros(w.shape) for w in self.weights]

        batch_outputs = []
        accuracy = 0
        losses = []
        for x, y in mini_batch:
            output, cache = self.__backprop(x, y)
            # 'inner' sum of weights and biases
            
            d_B = [local_db + update_db for local_db, update_db in zip(d_B, cache["dbs"])]
            d_W = [local_dw + update_dw for local_dw, update_dw in zip(d_W, cache["dws"])]

            batch_outputs.append(output)
            # accuracy - param with number of right predictions
            accuracy = self.__accuracy(output,y) +  accuracy
            loss = self.__cross_entropy_loss(output,y)
            losses.append(loss)

        batch_loss = sum(losses)/len(losses)
        res_accuracy = accuracy/len(list(mini_batch))
       
        temp_W = []
        temp_b = []  
        if len(self.losses)!=0 and batch_loss < 1.05 * self.losses[-1]:
            self.prev_dws = [self.momentum*prev_dw for prev_dw in self.prev_dws]
            self.prev_dbs = [self.momentum*prev_db for prev_db in self.prev_dbs]
            
        else:
            self.prev_dws = [np.zeros(w.shape) for w in self.weights]
            self.prev_dbs = [np.zeros(b.shape) for b in self.biases]
            
        for (w,dw,momentum) in zip(self.weights, d_W, self.prev_dws):
            temp_W.append(w-learning_rate/len(mini_batch)*dw + momentum)

        for (b,db,momentum) in zip(self.biases, d_B, self.prev_dbs):
            temp_b.append(b-learning_rate/len(mini_batch)*db + momentum)

        self.weights = temp_W
        self.biases = temp_b

        return batch_outputs,batch_loss,res_accuracy
                       
    def train(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        self.learing_rate = learning_rate
        training_data = training_data
        num_of_rows = len(training_data)
        self.batch_size = batch_size
        print("Started training \n")
        for k in range(epochs):
            # shuffling traing set before dividing to disjoint batches
            random.shuffle(training_data)
            mini_batches = [training_data[i:i+batch_size] for i in range(0, num_of_rows, batch_size)]
            batch_loss = [] 
            accuracies = []
            for mini_batch in mini_batches:
                batch_output, loss, mini_accuracy = self.__mini_batch_train(mini_batch, learning_rate)
                batch_loss.append(loss)

                accuracies.append(mini_accuracy)

            e_loss = sum(batch_loss)/len(batch_loss)
            print(f"Epoch {k}: acc: {sum(accuracies)/len(accuracies)} loss: {e_loss}\n")
            # mean loss for one epoch
            self.losses.append(e_loss)
        
        self.__plot_loss()

    def test(self, test_data,batch_size):
        
        num_of_rows = len(test_data)
    
        random.shuffle(test_data)
        mini_batches = [test_data[i:i+batch_size] for i in range(0, num_of_rows, batch_size)]
        batch_loss = [] 
        accuracies = []
        for mini_batch in mini_batches:
            losses = []
            accuracy = 0
            for x,y in mini_batch:
                output, cache = self.__feed_forward_pass(x)
                accuracy = self.__accuracy(output,y) +  accuracy
                loss = self.__cross_entropy_loss(output,y)
                losses.append(loss)
            batch_loss.append(sum(losses)/batch_size)
            accuracies.append(accuracy/batch_size)
        
        
        print(f"Test results: \nAcc: {np.mean(accuracies)} Loss: {np.mean(batch_loss)}\n")
        self.__plot_test_loss(batch_loss)

    def __plot_test_loss(self, losses):
        plt.figure()
        plt.plot(np.arange(len(losses)), losses)
        plt.xlabel("Sample number")
        plt.ylabel("Loss")
        plt.title(f"Test ")
        # plt.savefig(f'wine_momenum_{self.momentum}test.png')
        plt.show()

    def __plot_loss(self):
        plt.figure()
        # length of loss = num of epochs
        plt.plot(np.arange(len(self.losses)), self.losses)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train")
        # plt.savefig(f'wine_momenum_{self.momentum}train.png')
        plt.show()
    
