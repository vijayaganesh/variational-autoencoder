import numpy as np
from keras import backend

def reparametrization(args):
    mean, log_var = args
    batch = backend.shape(mean)[0]
    dim = backend.int_shape(mean)[1]
    epsilon = backend.random_normal(shape=(batch, dim))
    return mean + backend.exp(0.5 * log_var) * epsilon