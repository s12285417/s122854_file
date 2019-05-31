import numpy as np
import scipy.special


# neural network class definition
class NeuralNetWork:

    # initialise the neural networks
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input,hidden,outputlayers
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j,where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes, self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes, self.hnodes))

        # learning rate
        self.lr=learningrate

        # activation function is the sigmiod function
        self.activation_function=lambda x:scipy.special.expit(x)
        pass

    # train the neural networks
    def train(self, inputs_list, targets_list):
        # convert inputs to 2d arrays
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        # calculate signals into output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target-actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # update the weights for the links between the hidden and iutput layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs=np.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs=np.dot(self.wih,inputs)
        # calculate the signals emerging from hidden_inputs
        hidden_outputs=self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs=np.dot(self.who,hidden_outputs)
        # calculate the signals emerging from output layer
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

    pass


# number of input, hidden and output nodes
input_nodes=784
hidden_nodes=100
output_nodes=10

# learning rate is 0.3
learning_rate=0.2

# create instance of neural network
n=NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the mnist training data csv file into a list
train_data_file=open("mnist_data\mnist_train.csv","r")
train_data_list=train_data_file.readlines()
train_data_file.close()

pass

# train the neural network

# epochs is the number of times the training data set is used for training
epochs=5

for e in range(epochs):
    # go through the all records in the training data set
    for record in train_data_list:
        # spilt the record by ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets=np.zeros(output_nodes)+0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
pass

# load the mnist test data csv file into a list
test_data_file=open("mnist_data\mnist_test.csv","r")
test_data_list=test_data_file.readlines()
test_data_file.close()

# test neural network

# scorecard for how well the network performs, initially empty
scorecard=[]

# go though all the records in the test data set
for record in test_data_list:
    # split the records by the commas ','
    all_values=record.split(',')
    # correct answer is the first value
    correct_label=int(all_values[0])
    print(correct_label, 'correct label')
    # scale and shift the inputs
    inputs=(np.asfarray(all_values[1:])/255*0.99)+0.01
    # query the netwprk
    outputs=n.query(inputs)
    # the index of highest value correspond to the label
    label=np.argmax(outputs)
    print(label,"network's answer")
    # append correct or incorrect to lists
    if label==correct_label:
        # network's answer matches the correct answer,add 1 to the scorecard
        scorecard.append(1)
    else:
        # network's answer don't matches the correct answer,add 0 to the scorecard
        scorecard.append(0)
        pass
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array=np.asfarray(scorecard)
print('performance=',scorecard_array.sum()/scorecard_array.size)