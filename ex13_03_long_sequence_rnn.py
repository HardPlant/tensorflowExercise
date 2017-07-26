import tensorflow as tf
import numpy as np
sample = " if you want you"
idx2char = list(set(sample)) # index-> char, list(i,f,y,o,)
char2idx = {c:i for i, c in enumerate(idx2char)}

sample_idx = [char2idx[c] for c in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

#X data
X = tf.placeholder(tf.float32, [None, sequence_length])
#Y label
Y = tf.placeholder(tf.int32, [None, sequence_length])

num_classes = len(idx2char) # 10

X_one_hot = tf.one_hot(X,num_classes) # shape observation needed

cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell,X,initial_state=initial_state,dtype=tf.float32)

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
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        if i%10 == 0:
            print(i, "loss:", l, "prediction:", result, "true Y:", y_data)
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("\tPrediction str:", ''.join(result_str))
    