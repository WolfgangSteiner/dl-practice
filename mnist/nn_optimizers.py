import numpy as np

class Optimizer:
    def __init__(self, lr):
        self.lr = lr
    
    def update_step(self, X, grad, state):
        pass


class GradientDescent(Optimizer):
    def update_step(self, X, grad, state):
        return X - self.lr * grad


class MomentumGradientDescent(Optimizer):
    def __init__(self, lr, gamma=0.9):
        self.lr = lr
        self.gamma = 0.9

    def update_step(self, X, grad, state):
        v_prev = state.get('v', np.zeros_like(X))
        v = self.gamma * v_prev + self.lr * grad
        state['v'] = v
        return X - v


class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def update_step(self, X, grad, state):
        t = state.get('t', 1.0)
        m = state.get('m', np.zeros_like(X))
        v = state.get('v', np.zeros_like(X))

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * np.power(grad, 2)

        state['m'] = m
        state['v'] = v
        state['t'] = t + 1

        m_hat = m / (1 - self.beta1**t)
        v_hat = v / (1 - self.beta2**t)

        return X - self.lr / (np.sqrt(v_hat) + self.epsilon) * m_hat
