#! /usr/bin/python

import gzip
import struct
import numpy as np


def read_images(fn):
    data = gzip.open(fn, "rb").read()
    _, count, rows, columns = struct.unpack("!iiii", data[:16])
    return np.frombuffer(data[16:], dtype=np.uint8).reshape((count, rows*columns))


def read_labels(fn):
    data = gzip.open(fn, "rb").read()
    _, count = struct.unpack("!ii", data[:8])
    return np.frombuffer(data[8:], dtype=np.uint8).reshape((count, 1))


X_test = read_images("data/t10k-images-idx3-ubyte.gz")
X_train = read_images("data/train-images-idx3-ubyte.gz")
y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")
y_train = read_labels("data/train-labels-idx1-ubyte.gz")


assert X_test.shape[0] == y_test.shape[0]
assert X_train.shape[0] == y_train.shape[0]
