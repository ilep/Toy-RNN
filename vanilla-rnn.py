import numpy as np


def rnn_cell(h, x, wxh, whh, why):
    """
    Computes on step of the RNN.

    x: input vector (dimensions nx,1)
    h: hidden state vector (dimensions nh,1)
    wxh: input-hidden weights matrix (dimensions nx,nh)
    whh: hidden-hidden weights matrix (dimensions nh,nh)
    why: hidden-output weights matrix (dimensions nx,nh)
    y: output vector (dimensions nx,1)
    """
    h = np.tanh(whh * h + wxh * x)
    y = why * h
    return h, y

# create the dictionary
# perform one-hot encoding

# set hyper-parameters (hidden_size, learning_rate)
# random-initialize weights (wxh, whh, why)
# read N = size(X)
#for t in range(N):
#    h, y = rnn_cell(h, x, wxh, whh, why)
