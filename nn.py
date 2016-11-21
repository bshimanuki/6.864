import tensorflow as tf
import numpy as np
import embedding
import batch

batch_size = 10
lstm_size = 200 # TODO: allow lstm_size to be different from num_features
max_steps = 100
latent_dim = 2 * lstm_size
embedding_np = embedding.get_embedding()
num_words, num_features = embedding_np.shape

eos_embedding = embedding.get_eos_embedding()
eos_matrix = tf.reshape(tf.tile(tf.constant(
    eos_embedding, dtype=tf.float32),
    [batch_size]),
    [batch_size, 1, num_features])

# Placeholder for the inputs in a given iteration
# NOTE: words is padded! Never add eos to the end of words!
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
eos_plus_words = tf.reverse(tf.slice(tf.reverse(tf.concat(
    1, [eos_matrix, word_vectors]),
    [False, True, False]),
    [0, 1, 0], [-1, -1, -1]),
    [False, True, False])

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
mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
outputs_2D = tf.reshape(outputs, [-1, num_features])
logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
logits = tf.reshape(logits_2D, [batch_size, -1, num_words])
log_probs = tf.nn.log_softmax(logits)
unmasked_log_probs = tf.reduce_sum(tf.mul(
    log_probs, tf.one_hot(words, num_words)),
    reduction_indices=2)
batch_LL = tf.reduce_sum(tf.div(
    tf.mul(mask, unmasked_log_probs), tf.expand_dims(tf.reduce_sum(mask, reduction_indices=1), 1)),
    reduction_indices=1)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
loss = tf.reduce_mean(KLD - batch_LL)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

b = batch.Single()
# Execute some test code
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(embedding_init, feed_dict={embedding_placeholder:embedding_np})
    for i in range(1000):
        sentences, lengths = embedding.word_indices(b.next_batch(batch_size), eos=True)
        _, los = sess.run((train_step, loss), feed_dict={words:sentences, lens:lengths})
        print(los)
