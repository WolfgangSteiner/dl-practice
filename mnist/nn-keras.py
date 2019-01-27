from mnist import X_test, X_train, y_test, y_train
from utils import normalize, one_hot_encode, accuracy
import sklearn.model_selection
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, ELU, Softmax, BatchNormalization
from keras.optimizers import Adam

X_train, y_train = normalize(X_train), one_hot_encode(y_train)
X_test, y_test = normalize(X_test), one_hot_encode(y_test)

m = Sequential()
m.add(Dense(10, input_shape=(28*28,)))
m.add(ELU())
m.add(BatchNormalization())
m.add(Dense(10))
m.add(ELU())
m.add(BatchNormalization())
m.add(Dense(10))
m.add(Softmax())

optimizer = Adam(lr=0.001) 
m.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
m.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32) 

y_hat = m.predict(X_test)
print(f"accuracy: {accuracy(y_test, y_hat)}")


