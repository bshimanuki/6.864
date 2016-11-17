import tensorflow as tf

batch_size = 10
lstm_size = 10
max_steps = 100
num_features = 50
latent_dim = 2 * lstm_size

eos_matrix = tf.constant(np.zeros((batch_size, 1, num_features)))

# Placeholder for the inputs in a given iteration
words = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])
lens = tf.placeholder(tf.int32, [batch_size])
eos_plus_words = tf.concat(1, [eos_matrix, words])
lens_plus_one = tf.add(lens, 1)

# Construct LSTM
encoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)

# Get output and last hidden state
_, encoder_state = tf.nn.dynamic_rnn(encoder, words, sequence_length=lens, dtype=tf.float32)

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
decoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)

# Get output and last hidden state
outputs, _ = tf.nn.dynamic_rnn(encoder, eos_plus_words, sequence_length=lens_plus_one, dtype=tf.float32)
