import tensorflow as tf
import numpy as np
#shape(1,5,4) = [[[1,0,0,0(x4)],...(x5)]]
h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]

hidden_size = 2
sess = tf.Session()

#batching input
x_data = np.array([[h,e,l,l,o],
                   [e,o,l,l,l],
                   [l,l,e,e,l],], dtype= np.float32)
print(x_data.shape)
print(x_data)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                     state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5,3,4], 
    dtype=tf.float32)

sess.run(tf.global_variables_initializer())
#will be (1,5,2)
print(outputs.eval(session=sess))