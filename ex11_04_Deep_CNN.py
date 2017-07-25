import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
img = mnist.train.images[0].reshape(28,28)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1,28,28,1]) # n=-1, 28*28*color=1
Y = tf.placeholder(tf.float32, [None, 10])

#L1, ImgIn shape=(?,28,28,1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev = 0.01))
# conv -> (?, 28, 28,32) # 32 filters
# pool -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img,W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],
    strides=[1,2,2,1], padding='SAME')
#L2 ImgIn shape=(?,14,14,32)

W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))#64 filters
#conv -> (?, 14, 14, 64)
#pool -> (?,7,7,64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.reshape(L2, [-1,7*7*64]) # unfold, (?, 3136) <- (?, 784)

W3 = tf.get_variable("W2", shape=[7*7*64,10],
    initializer = tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf. random_normal([10])) # of out
hypothesis = tf.matmul(L2,W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#parameters
training_epochs = 15
batch_size = 100

print('Start')
training_epochs = 15
batch_size = 10
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _, = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c/total_batch
    print('Epoch:','%04d'%(epoch+1), 'cost=','{:.9f}'.format(avg_cost))

print('Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
    X:mnist.test.images,Y:mnist.test.labels}))



