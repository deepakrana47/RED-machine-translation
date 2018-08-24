import numpy as np
import warnings
warnings.filterwarnings("error")

def sigmoid(x):
    return np.power(1+np.exp(-x), -1)

def dsigmoid(x):
    t=sigmoid(x)
    return (1-t)*t

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return  1-np.square(np.tanh(x))

# Rectification linear activation function
def relu(x):
    return np.where(x < 0, 0, x)

# leaky Rectifing linear deactivation function
def drelu(x):
    return np.where(x > 0, 1, 0.01)

def etanh(x):
    return np.e * np.tanh( x / np.e)

def detanh(x):
    return 1-np.square(np.tanh(x)/np.e)

# Exponential linear dactivation function
def elu(x):
    # try:
        # return np.where(x > 0, x, np.exp(x) - np.ones(x.shape))
    return np.array([x[i] if x[i] > 0 else np.exp(x[i])-1 for i in range(len(x))])
    # except RuntimeWarning:
    #     print x
        # return np.where(x > 0, x, 0.0 - np.ones(x.shape))

# Exponential linear activation function
def delu(x):
    # return np.where(x > 0, np.ones(x.shape), np.exp(x))
    return np.array([[1.0] if x[i] > 0 else np.exp(x[i]) for i in range(len(x))])

# softmax activation function
def softmax(x):
    x = np.where(x > 50, 50, x)
    x = np.where(x < -50, -50, x)
    sum = np.sum(np.exp(x))
    t = np.exp(x) / sum
    return t

def dsoftmax(x):
    return 1

# Cross entropy loss function
def ce_erro(o, t):
    e = -np.sum(t * np.log(o))
    return e