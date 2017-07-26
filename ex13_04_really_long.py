import tensorflow as tf
import numpy as np
#hyper params
sentence = ("if you want to build a ship, don't drum up people together to"
            "collect wood and don't assign them task and work, but rather"
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence)) # index-> char, list(i,f,y,o,)
char_dic = {w:i for i, w in enumerate(char_set)}

dataX = []
dataY = []
#make datas
dic_size = len(char_dic) # RNN input size(one_hot)
rnn_hidden_size = len(char_dic) # RNN output size
num_classes = len(char_set) # 10, final output size
batch_size = 1 # one sentence data -> one batch
sequence_length = len(sentence) - 1 # LSTM unfolding unit

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.aapend(y)


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
        l, _ = sess.run([loss, train], feed_dict={X:dataX, Y:dataY})
        result = sess.run(prediction, feed_dict={X:dataX})
        
        print(i, "loss:", l, "prediction:", result, "true Y:", dataY)
        result_str = [char_set[c] for c in np.squeeze(result)]
        print("\tPrediction str:", ''.join(result_str))
