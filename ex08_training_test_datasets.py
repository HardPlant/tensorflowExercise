import tensorflow as tf
import code
x_data = [[1,2,1],[2,1,3],[3,1,3],[4,1,5],[1,7,5],[1,2,5],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0]]

x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]
assert(len(x_data) == len(y_data))
# one-hot encoding, only 1 in y

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([3, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.5).minimize(cost)

#prediction calculation
prediction = tf.argmax(hypothesis, 1) # pb -> one of o~6
correct_prediction = tf.equal(prediction, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, _ = sess.run([cost, W, optimizer],
             feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, cost_val, W_val)

    #predict
    print("Preiction :", sess.run(prediction, feed_dict = {X:x_test}))
    #calculate Accuracy
    print("Accuracy :", sess.run(accuracy, feed_dict = {X:x_test, Y:y_test}))

    #code.interact(local=locals())

