import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
img = mnist.train.images[0].reshape(28,28)

#for dropout
keep_prob = 0.7

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
#Let's drop out :)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))#64 filters
#conv -> (?, 14, 14, 64)
#pool -> (?,7,7,64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],
    padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))#64 filters
#conv -> (?, 14, 14, 64)
#pool -> (?,7,7,64)
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],
    padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1,128*4*4]) # unfold, (?, 3136) <- (?, 784)

W4 = tf.get_variable("W4", shape=[128*4*4,625],
    initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf. random_normal([625])) # of out
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[625, 10],
    initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf. random_normal([10])) # of out
hypothesis = tf.matmul(L4,W5) + b5

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
    X:mnist.test.images,Y:mnist.test.labels, keep_prob:1}))



