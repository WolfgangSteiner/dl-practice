import numpy as np


def softmax(x):
    exp_x = np.exp(x - x.max())
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


