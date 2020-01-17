from datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import random
import numpy.matlib
from utils import *

random.seed(1)
np.random.seed(1)

train_samples = 1500
val_samples = 500
test_samples = 1000

#loading train and test data
digits = list(range(10))
trX, trY, tsX, tsY, valX, valY = mnist(train_samples,val_samples,test_samples, digits=digits)

#visualizing few samples
plt.imshow(trX[:,0].reshape(28,28))
plt.imshow(tsX[:,100].reshape(28,28))
plt.imshow(valX[:,0].reshape(28,28))

#Testing parameter initialization
net_dims_tst = [5,4,1]
parameters_tst = initialize(net_dims_tst)
assert parameters_tst['W1'].shape == (4,5)
assert parameters_tst['b1'].shape == (4,1)

#Testing relu activation function
z_tst = np.array([[-1,2],[3,-6]])
a_tst, c_tst = relu(z_tst)
npt.assert_array_equal(a_tst,[[0,2],[3,0]])
assert (c_tst["Z"] == np.array([[-1,2],[3,-6]])).all()

#Testing gradient of relu activation function
dA_tst = np.array([[-7,5],[2,-3]])
cache_tst ={}
cache_tst["Z"] = np.array([[-1,1],[0,-3]])
dZ_tst = relu_der(dA_tst,cache_tst)
npt.assert_array_equal(dZ_tst,np.array([[0,5],[2,0]]))

#Testing softmax cross entropy loss
A_tst, _ ,_ = softmax_cross_entropy_loss(np.array([[-1,0,1],[2,1,-3]]))
npt.assert_almost_equal(np.sum(A_tst),3,5)

#Testing softmax cross entropy loss derivative
c_tst = {'A':np.array([[0.4,0.4],[0.6,0.6]])}
Y_tst = np.array([[0,1],[1,0]])
npt.assert_array_almost_equal(softmax_cross_entropy_loss_der(Y_tst, c_tst),np.array([[0.2,-0.3],[-0.2,0.3]]))

#Testing forward dropout
random.seed(1)
np.random.seed(1)

x = np.random.rand(10,10)
drop_prob = 0.2
out, cache = dropout_forward(x, drop_prob, mode='train')
npt.assert_almost_equal(np.sum(out),45.704208,6)

out, cache = dropout_forward(x, drop_prob, mode='test')
npt.assert_almost_equal(np.sum(out), 48.587792, 6)

#Testing backward dropout
np.random.seed(1)
random.seed(1)

dout = np.random.rand(3,2)
mask = np.random.rand(3,2)
mode = 'test'
cache = (mode, mask)
dA = dropout_backward(cache, dout)
npt.assert_almost_equal(np.sum(dA),1.6788879311798277,6)

mode = 'train'
cache = (mode, mask)
dA = dropout_backward(cache, dout)
npt.assert_almost_equal(np.sum(dA),0.61432916214326,6)

#Testing multi-layer forward propagation
X_tst = np.array([[1,3,2,5],[2,4,-2,6]])
param_tst ={'W1':[1,2],'b1':1}
drop_prob = 0.33
mode = 'test'
AL_t,c_t= multi_layer_forward(X_tst, param_tst, drop_prob, mode)
npt.assert_array_almost_equal(AL_t,np.array([ 6, 12, -1, 18]))

#Testing one layer backward propagation
cache_test ={}
A_tst = np.array([[1,3,2,5],[2,4,-2,6]])
cache_test['A'] = A_tst
W_tst = np.array([[3,2,1,-1],[1,3,1,3]])
b_tst = np.array([1,1]).reshape(-1,1)
dZ_tst = np.array([[0.1,0.3,0.1,.7],[.2,.3,0.3,.5]])

dA_prev,dW_tst,db_tst = linear_backward(dZ_tst, cache_test, W_tst, b_tst)
npt.assert_almost_equal(np.sum(dA_prev),16.4)
npt.assert_array_almost_equal(dW_tst,np.array([[4.7, 5.4],[4.2, 4. ]]))
npt.assert_array_almost_equal(db_tst,np.array([[1.2],[1.3]]))

#Testing classification
mode = 'train'
drop_prob = 0.2
X_tst = np.array([[-1,2,1],[1,1,3]])
param_tst ={'W1':5,'b1':3}
npt.assert_array_almost_equal(classify(X_tst, param_tst,mode,drop_prob),[[1,0,1]])

#Testing parameter updates with momentum
X_tst = [1,3,5,7]
param_tst ={'W1':5,'b1':7,'W2':2,'b2':3}
grad_tst ={'dW1':1,'db1':2,'dW2':-1,'db2':3}
epoch_tst = 1
v = {'dW1':1,'db1':1,'dW2':1,'db2':1}
learning_rate_tst = 1
decay_rate = 0.01
beta = 0.2

param_tst, al_tst, v_tst = update_parameters_with_momentum(param_tst, grad_tst, epoch_tst, v, beta, learning_rate_tst, decay_rate)

assert param_tst == {'W1': pytest.approx(4.009, 0.01), 'b1': pytest.approx(5.217, 0.01), 'W2': pytest.approx(2.594, 0.01), 'b2': pytest.approx(0.425, 0.01)}

#Training
# Configuration 1 - Overfittting case, No dropout regularization

net_dims = [784,516,256]
net_dims.append(10) # Adding the digits layer with dimensionality = 10
print("Network dimensions are:" + str(net_dims))

# getting the subset dataset from MNIST
train_data, train_label, test_data, test_label, val_data, val_label = mnist(noTrSamples=train_samples,noValSamples= val_samples,noTsSamples=test_samples,digits= digits)

# initialize learning rate and num_iterations
learning_rate = .03
num_iterations = 500

drop_prob = 0
mode = 'train'

costs,val_costs, parameters = multi_layer_network(train_data, train_label,val_data, val_label, net_dims, drop_prob, mode, \
	num_iterations=num_iterations, learning_rate= learning_rate)

# compute the accuracy for training set and testing set

mode ='test'
train_Pred = classify(train_data, parameters,mode,drop_prob)
val_Pred = classify(val_data, parameters,mode,drop_prob)
test_Pred = classify(test_data, parameters,mode,drop_prob)
print(train_Pred.shape)


trAcc = ( 1 - np.count_nonzero(train_Pred - train_label ) / float(train_Pred.shape[1])) * 100 
valAcc = ( 1 - np.count_nonzero(val_Pred - val_label ) / float(val_Pred.shape[1])) * 100 
teAcc = ( 1 - np.count_nonzero(test_Pred - test_label ) / float(test_Pred.shape[1]) ) * 100
print("Accuracy for training set is {0:0.3f} %".format(trAcc))
print("Accuracy for validation set is {0:0.3f} %".format(valAcc))
print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

X = range(0,num_iterations,10)
plt.plot(X,costs)
plt.plot(X,val_costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend(['Training','Validation'])
plt.show()

# Configuration 2 - using dropout regularization

net_dims = [784,516,256]
net_dims.append(10) # Adding the digits layer with dimensionality = 10
print("Network dimensions are:" + str(net_dims))

# getting the subset dataset from MNIST
train_data, train_label, test_data, test_label, val_data, val_label = mnist(noTrSamples=train_samples,noValSamples= val_samples,noTsSamples=test_samples,digits= digits)

# initialize learning rate and num_iterations
learning_rate = .03
num_iterations = 500

drop_prob = 0.2
mode = 'train'

costs,val_costs, parameters = multi_layer_network(train_data, train_label,val_data, val_label, net_dims, drop_prob, mode, \
	num_iterations=num_iterations, learning_rate= learning_rate)

# compute the accuracy for training set and testing set
mode ='test'
train_Pred = classify(train_data, parameters,mode,drop_prob)
val_Pred = classify(val_data, parameters,mode,drop_prob)
test_Pred = classify(test_data, parameters,mode,drop_prob)
print(train_Pred.shape)


trAcc = ( 1 - np.count_nonzero(train_Pred - train_label ) / float(train_Pred.shape[1])) * 100 
valAcc = ( 1 - np.count_nonzero(val_Pred - val_label ) / float(val_Pred.shape[1])) * 100 
teAcc = ( 1 - np.count_nonzero(test_Pred - test_label ) / float(test_Pred.shape[1]) ) * 100
print("Accuracy for training set is {0:0.3f} %".format(trAcc))
print("Accuracy for validation set is {0:0.3f} %".format(valAcc))
print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

X = range(0,num_iterations,10)
plt.plot(X,costs)
plt.plot(X,val_costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend(['Training','Validation'])
plt.show()



