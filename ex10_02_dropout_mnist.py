import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
#will automatically download datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10
#dropout -> 0.7 for train // 1 for testing
keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

w1 = tf.get_variable("w1", shape=[784,256],\
initializer= tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))

layer1 = tf.nn.relu(tf.matmul(X,w1)+b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

w2 = tf.get_variable("w2", shape=[256,256],\
initializer= tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))

layer2 = tf.nn.relu(tf.matmul(layer1,w2)+b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

w3 = tf.get_variable("w3", shape=[256,256],\
initializer= tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))

layer3 = tf.nn.relu(tf.matmul(layer2,w3)+b3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

w4 = tf.get_variable("w4", shape=[256,nb_classes],\
initializer= tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([nb_classes]))

#hypothesis == softmax
hypothesis = tf.matmul(layer3,w4)+b4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

#Test model
is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={
                    X: batch_xs , Y: batch_ys, keep_prob:0.7})# train
            avg_cost += c / total_batch
        
        print('Epoch:', '%04d' % (epoch + 1),
                'cost:', '{:.9f}'.format(avg_cost))
    print("Learning finished")

    print("Accuracy:", accuracy.eval(session=sess, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob:1}))#test

    #Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction : ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob:1}))
    
    plt.imshow(
        mnist.test.images[r:r+1].reshape(28,28), #image pixel size
        cmap='Greys',
        interpolation='nearest')
    plt.show()
