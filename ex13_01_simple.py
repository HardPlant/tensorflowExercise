import tensorflow as tf
import numpy as np
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

hidden_size = 2
sess = tf.Session()

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

x_data = np.array([[h,e,l,l,o]], dtype= np.float32)
print(x_data.shape)
print(x_data)

outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

sess.run(tf.global_variables_initializer())
print(outputs.eval(session=sess))