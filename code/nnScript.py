import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle
import matplotlib.pyplot as plt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1.0 / (1.0 + np.exp(-z))# your code here


selected_feature = None
def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    global selected_feature

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    # Combine the data from all sets (train, validation, test) into one array
    print("Step 1: Combining train, validation, and test sets into a single dataset...")
    combined_data = np.array(np.vstack((train_data, validation_data, test_data)))
    print("Combined data shape: ", combined_data.shape)

    # Check for duplicate columns by comparing each column with the first one
    print("Step 2: Identifying duplicate columns...")
    is_duplicate_column = np.all(combined_data == combined_data[0, :], axis=0)
    print("Number of duplicate columns: ", np.sum(is_duplicate_column))
    print("Number of unique columns: ", combined_data.shape[1] - np.sum(is_duplicate_column))

    # Identify the indices of the non-duplicate columns
    print("Step 3: Collecting indices of non-duplicate columns...")
    non_duplicate_columns = np.where(is_duplicate_column == False)
    print("Selected ", len(non_duplicate_columns[0]), " unique features.")

    # Remove duplicate columns from the combined dataset
    print("Step 4: Removing duplicate columns from the combined data...")
    cleaned_data = combined_data[:, ~is_duplicate_column]
    print("Cleaned data shape: ", cleaned_data.shape)

    # Store the indices of the selected features (non-duplicate columns)
    selected_feature = np.array([])
    selected_feature = non_duplicate_columns[0]
    print("Step 5: Feature selection completed.")

    # Re-split the cleaned data back into train, validation, and test sets
    print("Step 6: Re-splitting cleaned data back into train, validation, and test sets...")
    train_data = cleaned_data[:len(train_data), :]
    validation_data = cleaned_data[len(train_data):len(train_data) + len(validation_data), :]
    test_data = cleaned_data[(len(train_data) + len(validation_data)): (len(train_data) + len(validation_data) + len(test_data)),:]

    print("Final train set shape: ", train_data.shape)
    print("Final validation set shape: ", validation_data.shape)
    print("Final test set shape: ", test_data.shape)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    # Number of training samples
    num_samples = training_data.shape[0]
    
    # Feed forward - input layer to hidden layer
    w1_transposed = np.transpose(w1)
    w2_transposed = np.transpose(w2)
    
    # Adding bias term to the input layer
    bias_input = np.ones(shape=(num_samples, 1), dtype = np.float64)
    extended_train_data = np.append(training_data, bias_input, axis =1)  
    
    # Calculating activations for the hidden layer
    hidden_layer_input = np.dot(extended_train_data, w1_transposed) 
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Adding bias term to the hidden layer
    bias_hidden = np.ones((hidden_layer_output.shape[0], 1), dtype=np.float64)
    extended_hidden_output = np.append(hidden_layer_output, bias_hidden, axis =1)
    
    # Calculating output layer activations
    output_layer_input = np.dot(extended_hidden_output, w2_transposed)
    output_layer_output = sigmoid(output_layer_input)
    
    # Create the one-hot encoded labels for training
    one_hot_labels = np.zeros((num_samples, 10), dtype=np.float64)
    for i in range(one_hot_labels.shape[0]):   
        for j in range(one_hot_labels.shape[1]):
            if j==training_label[i]:
                one_hot_labels[i][j] = 1.0    
    
    # Calculate the error
    cross_entropy_loss = np.sum(one_hot_labels * np.log(output_layer_output) + (1 - one_hot_labels) * np.log(1 - output_layer_output))
    total_error = -cross_entropy_loss / num_samples
    
    # Regularization of error function
    w1_squared_sum = np.sum(np.square(w1))
    w2_squared_sum = np.sum(np.square(w2))
    regularization_term = (w1_squared_sum + w2_squared_sum) * lambdaval / (2 * num_samples)
    
    regularization_error = total_error + regularization_term 
    obj_val = regularization_error

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    # Backpropagation - Compute gradients for w1 and w2
    delta_output = output_layer_output - one_hot_labels 
    delta_output_transposed = np.transpose(delta_output)
    
    # Gradient of the weights for the output layer (w2)
    grad_w2 = np.dot(delta_output_transposed, extended_hidden_output)
    
    # Backpropagate to the hidden layer
    delta_hidden = np.dot(delta_output, w2[:, :-1]) * (1 - extended_hidden_output[:, :-1]) * extended_hidden_output[:, :-1]
    grad_w1 = np.dot(np.transpose(delta_hidden), extended_train_data)
    
    # Regularization of gradients
    grad_w2_regularized = (grad_w2 + lambdaval * w2) / num_samples
    grad_w1_regularized = (grad_w1 + lambdaval * w1) / num_samples
    
    # Combine the gradients into a single 1D array
    obj_grad = np.concatenate((grad_w1_regularized.flatten(), grad_w2_regularized.flatten()))
    

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.zeros(shape=(data.shape[0], 1))
    
    # Add bias term to input data
    bias_input = np.ones(shape=(data.shape[0],1), dtype=np.float64)
    extended_data = np.append(data, bias_input, axis=1) 
    
    # Compute activations for the hidden layer
    w1_transpose = np.transpose(w1)
    hidden_layer_input = np.dot(extended_data, w1_transpose) 
    hidden_layer_output = sigmoid(hidden_layer_input) 
    
    # Add bias term to the hidden layer output
    bias_hidden = np.ones((hidden_layer_output.shape[0], 1), dtype=np.float64)
    extended_hidden_output = np.append(hidden_layer_output, bias_hidden, axis=1) 
    
    # Compute activations for the output layer
    w2_transpose = np.transpose(w2)
    output_layer_input = np.dot(extended_hidden_output, w2_transpose)
    output_layer_output = sigmoid(output_layer_input)  
    
    # Predict the labels by choosing the class with the highest probability
    for i in range(output_layer_output.shape[0]):
        labels[i] = np.argmax(output_layer_output[i])
    

    return labels


def experiment_with_lambda_and_hidden_units(train_data, train_label, test_data, test_label, validation_data, validation_label, n_input, n_class):
    """ Experiment with different numbers of hidden units and lambda values """
    
    # Different values of hidden units and lambda to test
    hidden_units_list = [4,8,12,16,20]
    lambdas = np.arange(0, 61, 10)  # From 0 to 60 with step of 10
    
    training_times = []
    training_accuracies = []
    test_accuracies = []
    validation_accuracies = []
    
    for n_hidden in hidden_units_list:
        for lambdaval in lambdas:
            print(f"\n\nExperimenting with {n_hidden} hidden units and lambda={lambdaval}...")
            
            # Initialize weights
            initial_w1 = initializeWeights(n_input, n_hidden)
            initial_w2 = initializeWeights(n_hidden, n_class)
            initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
            
            # Arguments for nnObjFunction
            args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
            
            # Start training the model and record the time
            t1 = time.time()
            
            # Minimize the objective function
            nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options={'maxiter': 50})
            
            
            # Reshape nn_params to get weights w1 and w2
            w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
            w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
            
            # Calculate accuracy on training set
            predicted_label_train = nnPredict(w1, w2, train_data)
            training_accuracy = 100 * np.mean((predicted_label_train == train_label.reshape(train_label.shape[0], 1)).astype(float))
            training_accuracies.append(training_accuracy)

            t2 = time.time()
            
            # Record training time
            training_time = t2 - t1
            training_times.append(training_time)
            
            # Calculate accuracy on test set
            predicted_label_test = nnPredict(w1, w2, test_data)
            test_accuracy = 100 * np.mean((predicted_label_test == test_label.reshape(test_label.shape[0], 1)).astype(float))
            test_accuracies.append(test_accuracy)
            
            # Calculate accuracy on validation set
            predicted_label_validation = nnPredict(w1, w2, validation_data)
            validation_accuracy = 100 * np.mean((predicted_label_validation == validation_label.reshape(validation_label.shape[0], 1)).astype(float))
            validation_accuracies.append(validation_accuracy)

            
            print("Training accuracy: ", training_accuracy, "%")
            print("Testing accuracy: ", test_accuracy, "%")
            print("Validation accuracy: ", validation_accuracy, "%")
            print('Time taken:',training_time)
    
    # Reshape the accuracies to a 2D array for easier plotting
    training_accuracies = np.array(training_accuracies).reshape(len(hidden_units_list), len(lambdas))
    test_accuracies = np.array(test_accuracies).reshape(len(hidden_units_list), len(lambdas))
    validation_accuracies = np.array(validation_accuracies).reshape(len(hidden_units_list), len(lambdas))
    training_times = np.array(training_times).reshape(len(hidden_units_list), len(lambdas))
    
    # Plot the results
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))
    
    # Plot training time heatmap
    cax1 = axs[0].imshow(training_times, aspect='auto', cmap='coolwarm', origin='lower')
    axs[0].set_xticks(np.arange(len(lambdas)))
    axs[0].set_yticks(np.arange(len(hidden_units_list)))
    axs[0].set_xticklabels(lambdas)
    axs[0].set_yticklabels(hidden_units_list)
    axs[0].set_xlabel('Lambda')
    axs[0].set_ylabel('Hidden Units')
    axs[0].set_title('Training Time Heatmap')
    fig.colorbar(cax1, ax=axs[0])
    
    # Plot training accuracy heatmap
    cax2 = axs[1].imshow(training_accuracies, aspect='auto', cmap='coolwarm', origin='lower')
    axs[1].set_xticks(np.arange(len(lambdas)))
    axs[1].set_yticks(np.arange(len(hidden_units_list)))
    axs[1].set_xticklabels(lambdas)
    axs[1].set_yticklabels(hidden_units_list)
    axs[1].set_xlabel('Lambda')
    axs[1].set_ylabel('Hidden Units')
    axs[1].set_title('Training Accuracy Heatmap')
    fig.colorbar(cax2, ax=axs[1])
    
    # Plot test accuracy heatmap
    cax3 = axs[2].imshow(test_accuracies, aspect='auto', cmap='coolwarm', origin='lower')
    axs[2].set_xticks(np.arange(len(lambdas)))
    axs[2].set_yticks(np.arange(len(hidden_units_list)))
    axs[2].set_xticklabels(lambdas)
    axs[2].set_yticklabels(hidden_units_list)
    axs[2].set_xlabel('Lambda')
    axs[2].set_ylabel('Hidden Units')
    axs[2].set_title('Test Accuracy Heatmap')
    fig.colorbar(cax3, ax=axs[2])
    
    # Plot validation accuracy heatmap
    cax4 = axs[3].imshow(validation_accuracies, aspect='auto', cmap='coolwarm', origin='lower')
    axs[3].set_xticks(np.arange(len(lambdas)))
    axs[3].set_yticks(np.arange(len(hidden_units_list)))
    axs[3].set_xticklabels(lambdas)
    axs[3].set_yticklabels(hidden_units_list)
    axs[3].set_xlabel('Lambda')
    axs[3].set_ylabel('Hidden Units')
    axs[3].set_title('Validation Accuracy Heatmap')
    fig.colorbar(cax4, ax=axs[3])
    
    plt.tight_layout()
    plt.savefig('experiment_results_heatmap.png', dpi=300)
    plt.show(block=True)  # Ensures that the plot window stays open
    
    return hidden_units_list, lambdas, training_times, training_accuracies, test_accuracies, validation_accuracies




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
print(selected_feature)
print(len(selected_feature))

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in output unit
n_class = 10

#n_hidden_array = [4,8,12,16,20]

#for x in n_hidden_array:
#    for y in range(0,70,10):
        
# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = x # values - 4,8,12,16,20
n_hidden = 50 # Optimal hidden units

# set the regularization hyper-parameter
#lambdaval = y # values - 0 to 60, 10
lambdaval = 0

# Note current time
t1 = time.time()

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

print("\n Weight vector w1:"+str(w1))
print("\n Weight vector w2:"+str(w2))

# find the accuracy on Training Dataset

print('\n lambda:'+str(lambdaval)+'\n hidden layers:'+str(n_hidden))
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label.reshape(train_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label.reshape(validation_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label.reshape(test_label.shape[0], 1)).astype(float))) + '%')

t2 = time.time()

print('\n Time taken:'+str(t2-t1))


hidden_units_list, lambdas, training_times, training_accuracies, test_accuracies, validation_accuracies = experiment_with_lambda_and_hidden_units(
    train_data, train_label, test_data, test_label, validation_data, validation_label, n_input, n_class)


plt.figure(figsize=(18, 6))

# Plot training time vs lambda for each number of hidden units
plt.subplot(1, 4, 1)
for i, n_hidden in enumerate(hidden_units_list):
    plt.plot(lambdas, training_times[i, :], marker='o', label=f'{n_hidden} Hidden Units')
plt.xlabel('Lambda')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Lambda')
plt.legend()
plt.grid(True)

# Plot training accuracy vs lambda for each number of hidden units
plt.subplot(1, 4, 2)
for i, n_hidden in enumerate(hidden_units_list):
    plt.plot(lambdas, training_accuracies[i, :], marker='o', label=f'{n_hidden} Hidden Units')
plt.xlabel('Lambda')
plt.ylabel('Training Accuracy (%)')
plt.title('Training Accuracy vs Lambda')
plt.legend()
plt.grid(True)

# Plot test accuracy vs lambda for each number of hidden units
plt.subplot(1, 4, 3)
for i, n_hidden in enumerate(hidden_units_list):
    plt.plot(lambdas, test_accuracies[i, :], marker='o', label=f'{n_hidden} Hidden Units')
plt.xlabel('Lambda')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs Lambda')
plt.legend()
plt.grid(True)

# Plot validation accuracy vs lambda for each number of hidden units
plt.subplot(1, 4, 4)
for i, n_hidden in enumerate(hidden_units_list):
    plt.plot(lambdas, validation_accuracies[i, :], marker='o', label=f'{n_hidden} Hidden Units')
plt.xlabel('Lambda')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy vs Lambda')
plt.legend()
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('experiment_results_line.png', dpi=300)
plt.show()

# The choice of 20 hidden units and λ=10 is based on their ability to strike a balance between model complexity and generalization. 


# set the number of nodes in hidden unit (not including bias unit)
#n_hidden = x # values - 4,8,12,16,20
n_hidden = 20 # Optimal hidden units

# set the regularization hyper-parameter
#lambdaval = y # values - 0 to 60, 10
lambdaval = 10 # Optimal lambda value

# Note current time
t1 = time.time()

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

print("\n Weight vector w1:"+str(w1))
print("\n Weight vector w2:"+str(w2))

# find the accuracy on Training Dataset

print('\n lambda:'+str(lambdaval)+'\n hidden layers:'+str(n_hidden))
print('\n Training set Accuracy with optimal values of number of hidden units and lambda:' + str(100 * np.mean((predicted_label == train_label.reshape(train_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy with optimal values of number of hidden units and lambda:' + str(100 * np.mean((predicted_label == validation_label.reshape(validation_label.shape[0], 1)).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy with optimal values of number of hidden units and lambda:' + str(100 * np.mean((predicted_label == test_label.reshape(test_label.shape[0], 1)).astype(float))) + '%')

t2 = time.time()

print('\n Time taken:'+str(t2-t1))

# create params.picke file to store the with optimal values of number of hidden units and lambda

store_obj = dict([("selected_features", selected_feature),("w1",w1), ("w2",w2), ("n_hidden",n_hidden), ("lambdaval", lambdaval)])
pickle.dump(store_obj, open('params.pickle','wb'), protocol=3)