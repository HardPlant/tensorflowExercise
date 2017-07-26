import tensorflow as tf
import numpy as np

#input = 5 (demension(hello))
#seq = 6(len(hihello) -1 )
#hidden(output_demension) = 5 (one hot)
#batch = 1(string.num)

hidden_size = 5
input_dim = 5
batch_size = 1
sequence_length = 6

idx2char = ['h','i','e','l','o']
x_data = [[0,1,0,2,3,3]] # hihell
x_one_hot = [[[1,0,0,0,0], #h 0
              [0,1,0,0,0], #i 1
              [1,0,0,0,0], #h 0
              [0,0,1,0,0], #e 2
              [0,0,0,1,0], #l 3
              [0,0,0,1,0] #l 3
              ]]
y_data = [[1,0,2,3,3,4]] # ihello

#X one-hot, hidden_size = input_demension
X = tf.placeholder(tf.float32, [None, sequence_length,input_dim])
#Y label
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell,X,initial_state=initial_state,dtype=tf.float32)

y_data = tf.constant([[1,1,1]])
prediction = tf.constant([[[0.2,0.7],[0.6,0.2],[0.2,0.9]]], dtype=tf.float32)

weights = tf.constant([[1,1,1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=prediction, targets=y_data, weights=weights)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Loss: ", sequence_loss.eval())