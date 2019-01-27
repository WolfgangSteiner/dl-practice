import numpy as np
import pickle
from nn_activations import softmax


def exponential_decay(x_total, x_new, a):
    return x_total * a + x_new * (1 - a) 


class Layer:
    def forward(self, X, inference=False):
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
        self.w_state = {}
        self.b_state = {}

    def forward(self, X, inference=False):
        self.X = X
        return X@self.w + self.b
        
    def backprop(self, dy):
        self.dw = self.X.T@dy + self.l2 * self.w
        self.db = np.mean(dy, axis=0, keepdims=True)
        return dy@self.w.T

    def update(self, optimizer):
        self.w = optimizer.update_step(self.w, self.dw, self.w_state)
        self.b = optimizer.update_step(self.b, self.db, self.b_state)

    def clean(self):
        self.dw = None
        self.db = None
        self.X = None


class ReLU(Layer):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def forward(self, X, inference=False):
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

    def forward(self, X, inference=False):
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
    def forward(self, X, inference=False):
        self.y = softmax(X)
        return self.y

    def backprop(self, y):
        return self.y - y

    def update(self, optimizer):
        pass
    
    def clean(self):
        self.y = None


class BatchNormalization(Layer):
    def __init__(self):
        self.mu = None
        self.var = None
        self.gamma = None
        self.beta = None
        self.epsilon = 1.0e-7
        self.gamma_state = {}
        self.beta_state = {}
        self.mu_total = 0.0
        self.var_total = 1.0
        self.momentum = 0.99

    def forward(self, x, inference=False):
        if inference:
            return self.gamma * (x - self.mu_total) / np.sqrt(self.var_total + self.epsilon) + self.beta
        
        if self.gamma is None:
            self.gamma = np.ones(x.shape[1])
            self.beta = np.zeros(x.shape[1])
        self.m = x.shape[0]
        alpha = self.m / (self.m - 1.0 + self.epsilon)
        self.mu = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.mu_total  = exponential_decay(self.mu_total, self.mu, self.momentum)
        self.var_total = exponential_decay(self.var_total, self.var * alpha, self.momentum)
        self.x = x
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.epsilon)
        y = self.gamma * self.x_hat + self.beta
        return y

    def backprop(self, dy):
        self.dx_hat = self.gamma * dy
        self.d_var = -0.5 * np.sum(self.dx_hat * (self.x - self.mu), axis=0) \
                          * np.power(self.var + self.epsilon, -1.5)
        self.d_mu  = -1.0 * np.sum(self.dx_hat, axis=0) * np.power(self.var + self.epsilon, -0.5) \
                     - 2.0 * self.d_var * np.mean(self.x - self.mu, axis=0)
        self.dx = self.dx_hat * np.power(self.var + self.epsilon, -0.5) \
                + self.d_var * 2.0 * (self.x - self.mu) / self.m \
                + self.d_mu / self.m
        self.d_gamma = np.sum(dy * self.x_hat, axis=0)
        self.d_beta = np.sum(dy, axis=0)
        return self.dx

    def update(self, optimizer):
        self.gamma = optimizer.update_step(self.gamma, self.d_gamma, self.gamma_state)
        self.beta  = optimizer.update_step(self.beta, self.d_beta, self.beta_state)

