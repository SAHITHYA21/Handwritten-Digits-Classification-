'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle


# Do not change this
def initializeWeights(n_in, n_out):
    """
    # initializeWeights returns random weights for Neural Network given the
    # number of nodes in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))
    """
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    num_samples = train_data.shape[0]

    # Adding bias term to the input layer
    bias_input = np.ones(shape=(num_samples, 1), dtype=np.float64)
    extended_train_data = np.append(train_data, bias_input, axis=1)
    
    # Hidden layer computations
    hidden_layer_input = np.dot(extended_train_data, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Adding bias term to the hidden layer
    bias_hidden = np.ones((hidden_layer_output.shape[0], 1), dtype=np.float64)
    extended_hidden_output = np.append(hidden_layer_output, bias_hidden, axis=1)
    
    # Output layer computations
    output_layer_input = np.dot(extended_hidden_output, w2.T)
    output_layer_output = sigmoid(output_layer_input)
    
    # Step 2: One-hot encoding for the labels
    one_hot_labels = np.zeros((num_samples, n_class), dtype=np.float64)
    for i in range(one_hot_labels.shape[0]):
        one_hot_labels[i][int(train_label[i])] = 1.0
    
    # Step 3: Calculate the error
    cross_entropy_loss = np.sum(one_hot_labels * np.log(output_layer_output) + (1 - one_hot_labels) * np.log(1 - output_layer_output))
    total_error = -cross_entropy_loss / num_samples
    
    # Step 4: Regularization of error function
    regularization_term = (np.sum(np.square(w1)) + np.sum(np.square(w2))) * lambdaval / (2 * num_samples)
    obj_val = total_error + regularization_term

    # Backpropagation to compute gradients for w1 and w2
    delta_output = output_layer_output - one_hot_labels
    grad_w2 = np.dot(delta_output.T, extended_hidden_output) / num_samples + (lambdaval * w2) / num_samples
    
    delta_hidden = np.dot(delta_output, w2[:, :-1]) * (1 - hidden_layer_output) * hidden_layer_output
    grad_w1 = np.dot(delta_hidden.T, extended_train_data) / num_samples + (lambdaval * w1) / num_samples
    
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1, w2, data):
    labels = np.zeros(data.shape[0])
    
    # Add bias to input data
    bias_input = np.ones(shape=(data.shape[0], 1), dtype=np.float64)
    extended_data = np.append(data, bias_input, axis=1)
    
    # Hidden layer activations
    hidden_layer_input = np.dot(extended_data, w1.T)
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Add bias to hidden layer output
    bias_hidden = np.ones((hidden_layer_output.shape[0], 1), dtype=np.float64)
    extended_hidden_output = np.append(hidden_layer_output, bias_hidden, axis=1)
    
    # Output layer activations
    output_layer_input = np.dot(extended_hidden_output, w2.T)
    output_layer_output = sigmoid(output_layer_input)
    
    labels = np.argmax(output_layer_output, axis=1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# Unroll weights into a single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# Set regularization parameter
lambdaval = 10
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network
opts = {'maxiter': 50}
now = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
params = nn_params.get('x')

# Reshape parameters into w1 and w2
w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

print("Time to train:", time.time() - now)

# Test the model
predicted_label = nnPredict(w1, w2, train_data)
print('\nTraining set Accuracy:'+ str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)
print('\nValidation set Accuracy:'+ str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)
print('\nTest set Accuracy:'+ str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
print("Time on testing data:", time.time() - now)
