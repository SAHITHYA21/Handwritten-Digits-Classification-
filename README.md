# Handwritten-Digits-Classification

This repository contains a Python script (`nnScript.py`) that implements a neural network from scratch for classifying handwritten digits using the MNIST dataset.

## Overview

The script is organized into several functions, each responsible for critical aspects of the data preprocessing, model training, and evaluation workflow. It demonstrates the fundamentals of building and training a neural network without relying on external deep learning libraries.

## Functions

### 1. `preprocess()`
- **Description**: Loads and preprocesses the MNIST dataset.
- **Key Features**:
  - Loads `mnist_all.mat` file.
  - Splits data into training, validation, and test sets.
  - Normalizes pixel values to [0, 1].
  - Removes duplicate features to optimize performance.

### 2. `initializeWeights(input_size, output_size)`
- **Description**: Initializes weights and biases using a random uniform distribution.
- **Parameters**:
  - `input_size`: Number of input features.
  - `output_size`: Number of output nodes.
- **Returns**: A dictionary with initialized weight and bias matrices.

### 3. `sigmoid(z)`
- **Description**: Computes the sigmoid activation function.
- **Parameters**:
  - `z`: Input value or matrix.
- **Returns**: Sigmoid-transformed value.

### 4. `nnObjFunction(params, *args)`
- **Description**: Defines the objective function for the neural network, including forward pass, backpropagation, and cost calculation with regularization.
- **Parameters**:
  - `params`: Flattened weights and bias values.
  - `*args`: Includes training data, labels, and hyperparameters.
- **Returns**: Computed cost and gradients for optimization.

### 5. `nnPredict(W1, W2, data)`
- **Description**: Predicts class labels for the input dataset using the trained network.
- **Parameters**:
  - `W1, W2`: Neural network weights.
  - `data`: Input data.
- **Returns**: Predicted labels.

### 6. `train()`
- **Description**: Trains the neural network by optimizing weights using the L-BFGS-B method.
- **Key Features**:
  - Uses `nnObjFunction` for gradient computation.
  - Optimizes weights with `scipy.optimize.minimize`.

### 7. `evaluateModel(W1, W2, data, labels)`
- **Description**: Evaluates the trained neural network by calculating accuracy on a given dataset.
- **Parameters**:
  - `W1, W2`: Trained weights.
  - `data`: Dataset for evaluation.
  - `labels`: True labels for the dataset.
- **Returns**: Model accuracy on the dataset.

## How to Run

1. **Preprocess Data**: The `preprocess()` function prepares the MNIST dataset for training and testing.
2. **Train the Model**: Use the `train()` function to train the neural network.
3. **Evaluate the Model**: Test the trained model using the `evaluateModel()` function.

## Dependencies

Ensure the following Python libraries are installed:
- `numpy`
- `scipy`
- `matplotlib`

## Dataset

The script uses the MNIST dataset, which should be provided in the `mnist_all.mat` file format.
