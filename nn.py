import tensorflow as tf
import numpy as np
import data

batch_size = 10
lstm_size = 50 # TODO: allow lstm_size to be different from num_features
max_steps = 100
latent_dim = 2 * lstm_size
embedding_np = data.get_embedding()
num_words, num_features = embedding_np.shape

# TODO: Check that this is a trainable parameter
eos_matrix = tf.Variable(np.zeros((batch_size, 1, num_features)), dtype=tf.float32)

# Placeholder for the inputs in a given iteration
words = tf.placeholder(tf.int32, [batch_size, None])
lens = tf.placeholder(tf.int32, [batch_size])

lens_plus_one = tf.add(lens, 1)

# Construct LSTM
# TODO: Check that this is normalized
embedding_matrix = tf.Variable(tf.constant(0.0, shape=[num_words, num_features]),
                        trainable=False, name="embedding_matrix")
embedding_placeholder = tf.placeholder(tf.float32, [num_words, num_features])
embedding_init = embedding_matrix.assign(embedding_placeholder)

word_vectors = tf.nn.embedding_lookup([embedding_matrix], words)
eos_plus_words = tf.concat(1, [eos_matrix, word_vectors])

with tf.variable_scope('encoder'):
    encoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)

    # Get output and last hidden state
    _, encoder_state = tf.nn.dynamic_rnn(encoder, word_vectors, sequence_length=lens, dtype=tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.0001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

# Mu encoder
W_encoder_hidden_mu = weight_variable([2*lstm_size,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
mu_encoder = tf.matmul(encoder_state, W_encoder_hidden_mu) + b_encoder_hidden_mu

# Sigma encoder
W_encoder_hidden_logvar = weight_variable([2*lstm_size,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
logvar_encoder = tf.matmul(encoder_state, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.mul(std_encoder, epsilon)

# Decoder
with tf.variable_scope('decoder'):
    decoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)

    # Get output and last hidden state
    outputs, _ = tf.nn.dynamic_rnn(decoder, eos_plus_words, sequence_length=lens_plus_one, dtype=tf.float32)

# Compute probabilities
# TODO: outputs contains eos, but one_hot doesn't include eos
mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
outputs_reshaped = tf.reshape(outputs, [-1, num_features])
raw_probs = tf.matmul(outputs_reshaped, embedding_matrix, transpose_b=True)
logits = tf.reshape(raw_probs, [batch_size, -1, num_words])
log_probs = tf.nn.log_softmax(logits)
one_hot = tf.one_hot(words, num_words)
relevant_log_probs = tf.reduce_sum(tf.mul(log_probs, one_hot), reduction_indices=2)
losses = tf.div(tf.mul(mask, relevant_log_probs), tf.reduce_sum(mask, reduction_indices=1))
batch_losses = tf.reduce_sum(losses, reduction_indices=1)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
loss = tf.reduce_mean(KLD + batch_losses)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Execute some test code
with open('test_data.txt', 'r') as f:
    sentences = f.readlines()

sentences, lengths = data.word_indices(sentences)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(embedding_init, feed_dict={embedding_placeholder:embedding_np})
    sess.run(train_step, feed_dict={words:sentences, lens:lengths})
