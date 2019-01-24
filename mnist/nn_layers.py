import numpy as np
import pickle
from nn_activations import softmax

class Layer:
    def forward(self, X):
        return X

    def backprop(self, dy):
        return dy

    def update(self, optimizer):
        pass

    def clean(self):
        pass


class Dense(Layer):
    def __init__(self, n_in, n_out, l2=0.0):
        self.w = np.random.randn(n_in, n_out).astype(np.float32) / n_out
        self.b = np.zeros((1, n_out), dtype=np.float32)
        self.l2 = l2
        self.optimizer_state_w = {}
        self.optimizer_state_b = {}

    def forward(self, X):
        self.X = X
        return X@self.w + self.b
        
    def backprop(self, dy):
        self.dw = self.X.T@dy + self.l2 * self.w
        self.db = np.mean(dy, axis=0, keepdims=True)
        return dy@self.w.T

    def update(self, optimizer):
        self.w = optimizer.update_step(self.w, self.dw, self.optimizer_state_w)
        self.b = optimizer.update_step(self.b, self.db, self.optimizer_state_b)

    def clean(self):
        self.dw = None
        self.db = None
        self.X = None


class ReLU(Layer):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def forward(self, X):
        self.activation = X
        X[X <= 0.0] *= self.alpha
        return X

    def backprop(self, dy):
        dy[self.activation <= 0.0] *= self.alpha
        return dy

    def update(self, optimizer):
        pass

    def clean(self):
        self.activation = None


class ELU(Layer):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, X):
        self.activation = X
        X = np.maximum(X,0) + self.alpha * (np.exp(np.minimum(X, 0)) - 1)
        return X

    def backprop(self, dy):
        dy[self.activation < 0] *= self.alpha * np.exp(np.minimum(self.activation, 0))[self.activation<0]
        return dy

    def update(self, optimizer):
        pass

    def clean(self):
        self.activation = None


class SoftMax(Layer):
    def forward(self, X):
        self.y = softmax(X)
        return self.y

    def backprop(self, y):
        return self.y - y

    def update(self, optimizer):
        pass
    
    def clean(self):
        self.y = None


