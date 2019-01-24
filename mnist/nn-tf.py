from mnist import X_test, X_train, y_test, y_train
from utils import normalize, one_hot_encode, accuracy
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


X_train, y_train = normalize(X_train), one_hot_encode(y_train)
X_test, y_test = normalize(X_test), one_hot_encode(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


num_epochs = 10
batch_size = 32
num_batches = len(X_train) // batch_size + (len(X_train) % batch_size > 0)

x = tf.placeholder(dtype=tf.float32, shape=(None, 28*28))
y = tf.placeholder(dtype=tf.float32, shape=(None, 10))
lr = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(tf.random.normal(shape=(28*28, 300)))
b1 = tf.Variable(tf.random.normal(shape=(300,)))

w2 = tf.Variable(tf.random.normal(shape=(300, 10)))
b2 = tf.Variable(tf.random.normal(shape=(10,)))

z = tf.matmul(x, w1) + b1
z = tf.nn.elu(z)

logits = tf.matmul(z, w2) + b2
y_hat = tf.nn.softmax(logits)
loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)
accuracy_op = tf.reduce_mean(tf.to_float(tf.reduce_all(tf.equal(y, tf.round(y_hat)), axis=1)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}: ")
        loss = 0.0
        training_acc = 0.0
        for batch in range(num_batches):
            idx1 = batch * batch_size
            idx2 = min(len(X_train), idx1 + batch_size)
            r = range(idx1, idx2)
            X_batch = X_train[r, :]
            y_batch = y_train[r, :]
            current_loss, current_training_acc, _ = sess.run([loss_op, accuracy_op, train_op], feed_dict={lr:0.001, x:X_batch, y:y_batch})
            loss += current_loss
            training_acc += current_training_acc
            print("loss: %.4f,  acc: %4f" % (loss / (batch+1), training_acc / (batch+1)), end='\r')
        val_loss, val_acc = sess.run([loss_op, accuracy_op], feed_dict={x:X_val, y:y_val})
        print("validation loss: %.4f, validation accuracy: %4f" % (val_loss / len(X_val), val_acc))
