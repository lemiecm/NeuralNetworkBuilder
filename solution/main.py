from nn import *
from dataset_helpers import * 

if __name__ == "__main__":
    
    dataset_num = int(input("Choose dataset \n 1. Wine quality\n 2. Iris \n : "))
    no_neurons_input_layer = int(input("Enter number of neurons in input layer\n : "))
    no_of_hidden_layers = int(input("Enter number of hidden layers\n : "))
   
    no_of_neurons = []
    no_of_neurons.append(no_neurons_input_layer)

    for i in range(no_of_hidden_layers):
        neurons = int(input(f"Enter number of neurons for hidden layer #{i+1}\n : "))
        no_of_neurons.append(neurons)
    
    no_neurons_output_layer = int(input("Enter number of neurons for output layer\n : "))
    
    no_of_neurons.append(no_neurons_output_layer)

    num_activation_func = int(input("Choose activation function for layers \n 1.sigmoid \n 2.tanh \n : "))

    num_of_epochs = int(input("Enter number of epochs\n : "))

    batch_size = int(input("Enter batch size \n : "))

    learning_rate = float(input("Enter value for learning rate \n : "))

    momentum = float(input("Enter value for momentum \n : "))

    activation_func = 'sigmoid'
    if(num_activation_func==2):
        activation_func ='tanh'
    args = {}
    args["momentum"] = momentum
    args["activation_function"] = activation_func
    args['epochs'] = num_of_epochs
    args['batch_size'] = batch_size
    args["learning_rate"] = learning_rate
    # activation_func = 'sigmoid'
    # if(num_activation_func==2):
    #     activation_func =' tanh'
    
    # path = "./data/winequality-white.csv"
    # if(dataset_num == 2):
    #     path = "./data/wine.data"
    
    # lab_col = [11]
    # feat_col = [*range(0,11,1)]
    # if(dataset_num == 2):
    #     lab_col = 0
    
    
    # data = load_csv("./data/winequality-white.csv",[11],feat_col,is_header=True, is_csv =True)

    # train, test = spliting(data, num_of_input=no_neurons_input_layer, num_of_classes=11,label_col=11)

    # nn = NeuralNetwork([11,20,16,11])
    # nn.train(list(train),250,8,0.0001)
    run_from_app(dataset_num,no_of_neurons,args)