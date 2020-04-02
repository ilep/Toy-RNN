import numpy as np

def softmax(x):
    """
    Computes the softmax of the input vector.
    
    x: input vector (dimensions nx,1)
    """
    sum = np.sum(np.exp(x), axis=0)
    return np.exp(x) / sum

def rnn_cell(h, x, wxh, whh, why, bh, by):
    """
    Computes on step of the RNN.

    x: input vector (dimensions nx,1)
    h: hidden state vector (dimensions nh,1)
    wxh: input-hidden weights matrix (dimensions nx,nh)
    whh: hidden-hidden weights matrix (dimensions nh,nh)
    why: hidden-output weights matrix (dimensions nx,nh)
    bh: hidden bias vector (dimensions nh,1)
    by: output bias vector (dimensions nx,1)
    y: output vector (dimensions nx,1)
    """
    h = np.tanh(np.dot(whh,h) + np.dot(wxh,x) + bh)
    y = softmax(np.dot(why,h) + by)
    return h, y

# create the dictionary
# perform one-hot encoding

# set hyper-parameters (hidden_size, learning_rate)
# random-initialize weights (wxh, whh, why)
# read N = size(X)
#for t in range(N):
#    h, y = rnn_cell(h, x, wxh, whh, why)
