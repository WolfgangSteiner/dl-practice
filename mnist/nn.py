from mnist import X_test, X_train, y_test, y_train
from utils import normalize, one_hot_encode, accuracy
import numpy as np
import sys
import pickle
from nn_layers import Dense, ELU, ReLU, SoftMax
from nn_model import Model
from nn_activations import softmax
from nn_optimizers import GradientDescent, MomentumGradientDescent, Adam


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



if __name__ == "__main__":
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    nn = Model()
    nn.add(Dense(28*28, 10))
    nn.add(ELU())
    nn.add(Dense(10,10))
    nn.add(SoftMax())
    num_epochs = 20
    gen = Generator(X_train, y_train, 32)
    optimizer = Adam(0.001)

    try:
        for epoch in range(num_epochs):
            print("%03d: " % epoch, end="")
            for X_batch, y_batch in gen:
                y_hat = nn.forward(X_batch)
                nn.backprop(y_batch)
                nn.update(optimizer)
    
            print("training loss: %.4f" % loss(y_train, nn.forward(X_train)), end=", ")
            print("training acc : %.4f" % accuracy(y_train, nn.forward(X_train)))

        y_hat = nn.forward(X_test)
        print("Accuracy: %.4f" % accuracy(y_test, y_hat))

    except KeyboardInterrupt:
        nn.save("model.pickl")
        y_hat = nn.forward(X_test)
        print("Accuracy: %.4f" % accuracy(y_test, y_hat))




