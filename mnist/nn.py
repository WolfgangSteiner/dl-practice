from mnist import X_test, X_train, y_test, y_train
from utils import normalize, one_hot_encode, accuracy
import numpy as np
import sys
import pickle
from nn_layers import Dense, ELU, ReLU, SoftMax, BatchNormalization
from nn_model import Model
from nn_activations import softmax
from nn_optimizers import GradientDescent, MomentumGradientDescent, Adam
from sklearn.model_selection import train_test_split
import sklearn.utils

class Generator:
    def __init__(self, X, y, batch_size=16):
        self.X = X
        self.y = y
        self.idx = 0
        self.batch_size = batch_size

    def __next__(self):
        if self.idx < self.X.shape[0]:
            cur, self.idx = self.idx, min(self.idx + self.batch_size, self.X.shape[0])
            return self.X[cur:self.idx], self.y[cur:self.idx]
        else:
            self.idx = 0
            raise StopIteration()

    def __iter__(self):
        return self


def loss(y,y_hat):
    return -np.sum(y * np.log(y_hat + 1e-12), axis=1, keepdims=True).mean()


def print_status(epoch, X_train, y_train, X_val, y_val, update=False):
    s = "%03d:" % epoch 
    s += " train loss: %.4f" % loss(y_train, nn.predict(X_train))
    s += " train acc : %.4f" % accuracy(y_train, nn.predict(X_train))
    s += " val loss: %.4f" % loss(y_val, nn.predict(X_val))
    s += " val acc : %.4f" % accuracy(y_val, nn.predict(X_val))
    end = "\r" if update else "\n"
    print(s + (100 - len(s))*" ", end=end)


if __name__ == "__main__":
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

    nn = Model()
    nn.add(Dense(28*28, 10))
    nn.add(ELU())
    nn.add(BatchNormalization())
    nn.add(Dense(10,10))
    nn.add(ELU())
    nn.add(BatchNormalization())
    nn.add(Dense(10,10))
    nn.add(SoftMax())
    num_epochs = 20
    gen = Generator(X_train, y_train, 32)
    optimizer = Adam(0.001)

    try:
        for epoch in range(num_epochs):
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            batch = 0
            print("%03d: " % epoch, end="")
            for X_batch, y_batch in gen:
                y_hat = nn.forward(X_batch)
                nn.backprop(y_batch)
                nn.update(optimizer)
                if batch % 100 == 0:
                    print_status(epoch, X_train, y_train, X_val, y_val, update=True)
                    sys.stdout.flush()
                batch += 1

            print_status(epoch, X_train, y_train, X_val, y_val)


        y_hat = nn.predict(X_test)
        print("Accuracy: %.4f" % accuracy(y_test, y_hat))

    except KeyboardInterrupt:
        nn.save("model.pickl")
        y_hat = nn.predict(X_test)
        print("Accuracy: %.4f" % accuracy(y_test, y_hat))




