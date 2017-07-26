import tensorflow as tf
import numpy as np
sentence = ("if you want to build a ship, don't drum up people together to"
            "collect wood and don't assign them task and work, but rather"
            "teach them to long for the endless immensity of the sea.")
idx2char = list(set(sample)) # index-> char, list(i,f,y,o,)
char2idx = {c:i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
#hyper params
dic_size = len(char2idx) # RNN input size(one_hot)
rnn_hidden_size = len(char2idx) # RNN output size
num_classes = len(idx2char) # 10, final output size
batch_size = 1 # one sample data -> one batch
sequence_length = len(sample) - 1 # LSTM unfolding unit

#X data
X = tf.placeholder(tf.int32, [None, sequence_length])
#Y label
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X,num_classes) # shape observation needed

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell,X_one_hot,initial_state=initial_state,dtype=tf.float32)

#[batch_size * sequence_length]
weights = tf.ones([batch_size, sequence_length])

#output should not directly here, but for simple example
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)
#will optimize weights for logit function, target is Y
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_data})
        
        print(i, "loss:", l, "prediction:", result, "true Y:", y_data)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str:", ''.join(result_str))
