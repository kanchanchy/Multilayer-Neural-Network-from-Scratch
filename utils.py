from datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import random
import numpy.matlib

#parameter initialization
def initialize(net_dims):
    '''
    Inputs:
    net_dims - Array containing the dimensions of the network. 
    Outputs:
    parameters - Dictionary element for storing the Weights and bias of each layer of the network'''
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        weights = np.random.normal(size = (net_dims[l + 1], net_dims[l]))
    	bias = np.zeros((net_dims[l + 1], 1))
    	parameters["W"+str(l+1)] = weights
    	parameters["b"+str(l+1)] = bias
    	return parameters

#relu activation function implementation
def relu(Z):
    '''
    Computes relu activation of Z
    Inputs: 
        Z is a numpy.ndarray (n, m) which represent 'm' samples each of 'n' dimension    
    Returns: 
        A is activation. numpy.ndarray (n, m) representing 'm' samples each of 'n' dimension
        cache is a dictionary with {"Z", Z}'''
    cache = {}
    matrix_zero = np.zeros_like(Z)
    A = np.maximum(matrix_zero, Z)
    cache["Z"] = Z
    return A, cache

#computing gradient of relu activation function
def relu_der(dA, cache):
    '''
    Computes derivative of relu activation
    Inputs: 
        dA is the derivative from the upstream layer with dimensions (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation
    Returns: 
    dZ is the derivative. numpy.ndarray (n,m) '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ

#implementing linear activation function
def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness
    Inputs: 
        Z is a numpy.ndarray (n, m)
    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}'''
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache

#computing derivative of linear activation function
def linear_der(dA, cache):
    '''
    This function is implemented for completeness
    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation
    Returns: 
    dZ is the derivative. numpy.ndarray (n,m)'''      
    dZ = np.array(dA, copy=True)
    return dZ

#implementing softmax cross entropy loss
def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss
    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (d, m) - labels in one-hot representation
            when y=[] loss is set to []
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction'''
    #Calculating softmax activation
    n, m = Z.shape
    Z_trans = Z.T
    A_trans = []
    for i in range(m):
    	Z_max = np.max(Z_trans[i])
    	Z_exp = np.exp(Z_trans[i] - Z_max)
    	A_trans.append(Z_exp/np.sum(Z_exp))
    	A = np.array(A_trans).T

    #Calculating loss
    loss = []
    if len(Y) != 0:
    	A_trans = A.T
    	A_trans = np.where(A_trans == 0, 0.00000001, A_trans)
    	Y_trans = Y.T
    	loss = 0
    	for i in range(m):
    		log_probability = np.multiply(Y_trans[i], np.log(A_trans[i]))
    		loss += np.sum(log_probability)
        #print(loss)
        loss = (-1/m)*loss

        cache = {}
        cache["A"] = A
        return A, cache, loss

#One hot representation of classes
def one_hot(Y,num_classes):
    '''
    Return one hot vector for the lables
    Inputs:
        Y - Labels of dimension (1,m)
        num_classes - Number of output classes 
    Ouputs:
    Y_one_hot - one hot vector of dimension(n_classes,m)'''
    Y_one_hot = np.zeros((num_classes,Y.shape[1]))
    for i in range(Y.shape[1]):
    	Y_one_hot[int(Y[0,i]),i] = 1
    	return Y_one_hot

#derivative of softmax cross entropy loss
def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss
    Inputs: 
        Y - numpy.ndarray (n,m) is a one-hot encoding of the ground truth labels
        cache -  a dictionary with cached activations A of size (n,m)
    Returns:
    dZ - numpy.ndarray (n, m) derivative for the previous layer'''
    A = cache["A"]
    dZ = np.divide(np.subtract(A, Y), A.shape[1])
    return dZ

#implementing forward dropout
def dropout_forward(A, drop_prob, mode='train'):
    '''
    Using the 'inverted dropout' technique to implement dropout regularization.
    Inputs:
        A - Activation matrix
        drop_prob - the probability of dropping a neuron's activation.
                
        mode - Dropout acts differently in training and testing mode. Hence, mode is a parameter which
        takes in only 2 values, 'train' or 'test'
    Outputs:
        out - Output of shape(n,m) same as input but with some values masked out.
        cache - a tuple which stores the values that are required in the backward pass.'''
        mask = None
        out = None

    if mode == 'train':
        mask = np.random.rand(*A.shape) > drop_prob
        out = np.multiply(A, mask)
        out = np.divide(out, (1 - drop_prob))

    elif mode == 'test':
        out = np.array(A, copy=True)
    else:
        raise ValueError("Mode value not set, set it to 'train' or 'test'")
    cache = (mode, mask)
    out = out.astype(A.dtype, copy=False)
    return out, cache

#implementing backward
def dropout_backward(cache, dout):
    '''
    Backward pass for the inverted dropout.
    Inputs: 
        dout: derivatives from the upstream layers of dimension (n,m).
        cache: contains the mask, input, and chosen dropout probability from the forward pass.
    Outputs:
        dA = derivative from the layer of dimension (n,m)'''
    dA = None
    mode, mask = cache
    if mode == 'train':
        dA = np.multiply(dout, mask)
    elif mode == 'test':
        dA = np.array(dout, copy=True)
    else:
        raise ValueError("Mode value not set, set it to 'train' or 'test'")
    return dA

#forward propagation for one layer
def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 
    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
    Returns:
        out = dropout(WA + b), where out is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A'''
    Z = np.dot(W, A) + b

    cache = {}
    cache["A"] = A
    return Z, cache

#forward propagation with activation for one layer
def layer_forward(A_prev, W, b, activation, drop_prob, mode):
    '''
    Input A_prev propagates through the layer and the activation
    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function
    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative'''

    Z, lin_cache = linear_forward(A_prev, W, b)

    if activation == "relu":
        A, act_cache = relu(Z)
        A, drop_cache =  dropout_forward(A, drop_prob, mode)

    elif activation == "linear":
        A, act_cache = linear(Z)
        drop_cache = None


    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    cache["drop_cache"] = drop_cache

    return A, cache

#multi-layer forward propagation
def multi_layer_forward(X, parameters,drop_prob, mode):
    '''
    Forward propgation through the layers of the network
    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
        where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs'''
    L = len(parameters)//2  
    A = X
    caches = []

    for l in range(1,L):
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu", drop_prob, mode)
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear",drop_prob, mode)
    caches.append(cache)
    return AL, caches

#backward propagation for one layer
def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer
    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b'''      
    A = cache['A']
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A.T)
    db = np.sum(dZ, axis = 1, keepdims = True)
    return dA_prev, dW, db

#backward propagation for one layer with activation
def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer
    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b'''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    drop_cache = cache["drop_cache"]

    if activation == "relu":
        dA = dropout_backward(drop_cache, dA)
        dZ = relu_der(dA, act_cache)

    elif activation == "linear":
        dZ = linear_der(dA, act_cache)

    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

#multi-layer backward propagation
def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately
    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        gradients - dictionary of gradient of network parameters 
        {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}'''

    L = len(caches) 
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = layer_backward(dA, caches[l-1], parameters["W"+str(l)], parameters["b"+str(l)], activation)
        activation = "relu"
    return gradients

#prediction
def classify(X, parameters,mode,drop_prob):
    '''
    Network prediction for inputs X
    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
    YPred - numpy.ndarray (1,m) of predictions'''
    A_forward, cache_forward = multi_layer_forward(X, parameters, drop_prob, mode)
    A_softmax, cache_softmax, loss = softmax_cross_entropy_loss(A_forward)
    YPred = [np.argmax(A_softmax, axis=0)]
    return np.array(YPred)

#calculating momentum
def initialize_velocity(parameters):
    '''
    Inputs:
        parameters - The Weight and Bias parameters of the network 
    Outputs:
    v - velocity parameter'''
    L = len(parameters) // 2 
    v = {}
    
    # Initialize velocity
    for l in range(L):
    	v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l+1)])
    	v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l+1)])

    return v

#updating parameters with momentum
def update_parameters_with_momentum(parameters, gradients, epoch, v, beta, learning_rate, decay_rate=0.01):
    '''
    Updates the network parameters with gradient descent
    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        v - Velocity parameter
        beta - momentum parameter
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
        '''

    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters) // 2 # number of layers in the neural networks
    
    for i in range(L):
        #Calculation for weights
        V_weight = beta*v["dW" + str(i + 1)] + (1 - beta)*gradients["dW" + str(i + 1)]
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - alpha*V_weight
        
        #Calculation for bias
        V_bias = beta*v["db" + str(i + 1)] + (1 - beta)*gradients["db" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - alpha*V_bias
    return parameters, alpha, v

#implementing multi-layer neural networks
def multi_layer_network(X, Y,valX, valY, net_dims, drop_prob, mode, num_iterations=500, learning_rate=0.2, decay_rate=0.00005):
    '''
    Creates the multilayer network and trains the network
    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        valX - numpy.ndarray(n,m) of validation data
        valY - numpy.ndarray(1,m) of validation data labels
        net_dims - tuple of layer dimensions
        drop_prob - dropout parameter, we drop the number of neurons in a given layer with respect to prob.
        mode - Takes in 2 values 'train' or 'test' mode. Model behaviour is dependent on the mode.
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
        decay_rate - the rate at which the learning rate is decayed.
    Returns:
        costs - list of costs over training
        val_costs - list of validation costs over training
        parameters - dictionary of trained network parameters'''

    parameters = initialize(net_dims)
    A0 = X
    costs = []
    val_costs = []
    num_classes = 10
    Y_one_hot = one_hot(Y,num_classes)
    valY_one_hot = one_hot(valY,num_classes)
    alpha = learning_rate
    beta = 0.9
    for ii in range(num_iterations):
        #Forward Propagation with training data
        Z, cache_1 = multi_layer_forward(A0, parameters, drop_prob, mode)
        AL, cache_2, cost = softmax_cross_entropy_loss(Z, Y_one_hot)

        #Backward Propagation with training data
        dZ = softmax_cross_entropy_loss_der(Y_one_hot, cache_2)
        gradients = multi_layer_backward(dZ, cache_1, parameters)
        v = initialize_velocity(parameters)
        parameters, alpha, v = update_parameters_with_momentum(parameters, gradients, num_iterations, v, beta, learning_rate, decay_rate)
        
        #Forward Propagation with validation data
        Z_, cache_ = multi_layer_forward(valX, parameters, drop_prob, 'test')
        AL_, cache_val, val_cost = softmax_cross_entropy_loss(Z_, valY_one_hot)

        if ii % 10 == 0:
            costs.append(cost)
        	val_costs.append(val_cost)
        if ii % 10 == 0:
        	print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, alpha))

    return costs, val_costs, parameters