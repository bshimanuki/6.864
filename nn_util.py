import tensorflow as tf


def weight_variable(shape, name=None):
    with tf.name_scope('weight'):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.0001))
        tf.histogram_summary('%s/weight' % (name if name is not None else ''), weight)
        return weight


def bias_variable(shape, name=None):
    with tf.name_scope('bias'):
        bias = tf.Variable(tf.zeros(shape=shape))
        tf.histogram_summary('%s/bias' % (name if name is not None else ''), bias)
        return bias


def ff_layer(input_layer, w, b, name = None):
    with tf.name_scope(name):
        output = tf.matmul(input_layer, w) + b
        return output


def ff_layer_vars(input_depth, output_depth, name = None):
    """
    Used in a fully connected layer:

    (batch_size, input_depth) -> (batch_size, depth)
    :param input_depth:
    :param output_depth:
    :param name:
    :return:
    """
    with tf.name_scope(name):
        w = weight_variable([input_depth, output_depth], name)
        b = bias_variable([output_depth], name)
        return (w, b)


def encoder_layer(word_vectors, lens):
    with tf.name_scope('encoder_rnn'):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=word_vectors.get_shape().as_list()[-1], state_is_tuple=False)
        _, encoder_state = tf.nn.dynamic_rnn(lstm_cell, word_vectors, sequence_length=lens, dtype=tf.float32)
        return encoder_state


def sampling_layer(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')

    # Sample latent variable
    std_encoder = tf.exp(0.5 * logvar)
    z = mu + tf.mul(std_encoder, epsilon)
    return z


def decoder_layer(z, word_vectors, lens, eos_matrix):
    # TODO: potentially do not need to pass in word vectors / EOS matrix here?
    eos_plus_words = tf.reverse(tf.slice(tf.reverse(tf.concat(
        1, [eos_matrix, word_vectors]),
        [False, True, False]),
        [0, 1, 0], [-1, -1, -1]),
        [False, True, False])
    decoder = tf.nn.rnn_cell.LSTMCell(num_units=word_vectors.get_shape().as_list()[-1], state_is_tuple=False)
    outputs, _ = tf.nn.dynamic_rnn(decoder, eos_plus_words, sequence_length=lens+1, initial_state=z, dtype=tf.float32)
    return outputs