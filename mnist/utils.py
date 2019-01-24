import numpy as np

def normalize(x):
    return np.array((x - x.mean()) / x.std(), dtype=np.float32)

def one_hot_encode(x):
    result = np.zeros((x.shape[0], 10), dtype=np.float32)
    for i in range(10):
        c = (x == i).reshape(-1,1).astype(np.float32)
        result[:,i:i+1] = c
    return np.array(result, dtype=np.float32)


def accuracy(y, y_hat):
    return np.mean(np.all(y == y_hat.round(), axis=1))
