import tensorflow as tf
X = [1,2,3]
Y = [1,2,3]
W = tf.Variable(5.)
#Linear model
hypothesis = W*X
#Manual gradient
gradient = tf.reduce_mean((W*X-Y)*X) *2
#cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#Get gradients
gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)