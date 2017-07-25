import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        #for dropout
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1,28,28,1]) # n=-1, 28*28*color=1
            self.Y = tf.placeholder(tf.float32, [None, 10])
        
            self.keep_prob = tf.placeholder(tf.float32)
            
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
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))#64 filters
            #conv -> (?, 14, 14, 64)
            #pool -> (?,7,7,64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],
                padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

            W3 = tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))#64 filters
            #conv -> (?, 14, 14, 64)
            #pool -> (?,7,7,64)
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],
                padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            L3 = tf.reshape(L3, [-1,128*4*4]) # unfold, (?, 3136) <- (?, 784)

            W4 = tf.get_variable("W4", shape=[128*4*4,625],
                initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf. random_normal([625])) # of out
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            W5 = tf.get_variable("W5", shape=[625, 10],
                initializer = tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf. random_normal([10])) # of out
            self.hypothesis = tf.matmul(L4,W5) + b5

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).\
                    minimize(self.cost)

            self.correct_prediction = tf.equal(tf.argmax(self.hypothesis,1),
            tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
            tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.logits,
            feed_dict={self.X:x_test, self.keep_prob:keep_prop})
    
    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={ 
            self.X:x_test,self.Y:y_test,self.keep_prob:keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict =
         {self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
img = mnist.train.images[0].reshape(28,28)

sess = tf.Session()
m1 = Model(sess,"m1")
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
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c/total_batch
    print('Epoch:','%04d'%(epoch+1), 'cost=','{:.9f}'.format(avg_cost))

print('Finished!')

print('Accuracy:', m1.get_accuracy(mnist.test.images,mnist.test.labels))



