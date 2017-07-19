import tensorflow as tf

print("Hi!")

x_train = [1,2,3]
y_train = [1,2,3]
#H(x) = Wx+b
#tf Variable - used by Tensorflower, self-trainable
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis Wx+b
hypothesis = x_train * W + b

#cost/Loss function
#reduce_mean -> get mean
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
#goal -> minimize cost
#Minimize it (cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#Gradient -> Slope, Slope Descent-following / Multi-var(W(a,b,c,...)) appliable
train=optimizer.minimize(cost) # a node

#Launch the graph.
sess = tf.Session()
#init global variable
sess.run(tf.global_variables_initializer()) # variables -> W, b
#Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

