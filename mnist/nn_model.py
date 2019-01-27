import pickle


class Model:
    def __init__(self):
        self.layers = []

    def forward(self, X):
        for l in self.layers:
            X = l.forward(X)
        return X

    def predict(self, X):
        for l in self.layers:
            X = l.forward(X, inference=True)
        return X
    
    def backprop(self, dy):
        for l in reversed(self.layers):
            dy = l.backprop(dy)

    def update(self, optimizer):
        for l in self.layers:
            l.update(optimizer)

    def add(self, layer):
        self.layers.append(layer)

    def clean(self):
        for l in self.layers:
            l.clean()

    def save(self, fn):
        self.clean()
        with open(fn, "wb") as f:
            pickle.dump(self, f)

