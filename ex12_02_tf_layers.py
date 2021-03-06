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
        
            self.training = tf.placeholder(tf.bool)
            
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3],
                padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], padding='SAME', strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
            # training=false->rate will 1.0 (on test)

            
            #L2 ImgIn shape=(?,14,14,32)
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3],
               padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], padding='SAME', strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3],
               padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], padding='SAME', strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

            #flat & dense
            flat = tf.reshape(dropout3, [-1,128*4*4])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(dense4, rate=0.5, training=self.training)

            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).\
                minimize(self.cost)

        self.correct_prediction = tf.equal(tf.argmax(self.logits,1),
        tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
        tf.float32))

    def predict(self, x_test, training=1.0):
        return self.sess.run(self.logits,
            feed_dict={self.X:x_test, self.training:training})
    
    def get_accuracy(self, x_test, y_test, training=1.0):
        return self.sess.run(self.accuracy, feed_dict={ 
            self.X:x_test,self.Y:y_test,self.training:training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict =
         {self.X: x_data, self.Y: y_data, self.training: training})


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

print('started!')
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



