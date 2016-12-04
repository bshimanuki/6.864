import tensorflow as tf

from constants import BATCH_SIZE
import embedder


def weight_variable(shape, name=None):
    weight = tf.get_variable('weight', initializer=tf.truncated_normal(shape, stddev=0.0001))
    tf.histogram_summary('%s/weight' % (name if name is not None else ''), weight)
    return weight


def bias_variable(shape, name=None):
    bias = tf.get_variable('bias', initializer=tf.zeros(shape=shape))
    tf.histogram_summary('%s/bias' % (name if name is not None else ''), bias)
    return bias


def ff_layer(input_layer, name = None):
    with tf.variable_scope(name):
        output = tf.matmul(input_layer, tf.get_variable('weight')) + tf.get_variable('bias')
        return output


def init_ff_layer_vars(input_depth, output_depth, name = None):
    """
    Used in a fully connected layer:

    (batch_size, input_depth) -> (batch_size, depth)
    :param input_depth:
    :param output_depth:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        w = weight_variable([input_depth, output_depth], name)
        b = bias_variable([output_depth], name)


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


def hippopotamus(words, lens, num_features, num_words):
    with tf.variable_scope('embedding', reuse=True):
        embedding_matrix = tf.get_variable('embedding_matrix')
        eos_matrix = tf.get_variable('eos_matrix')

    word_vectors = tf.nn.embedding_lookup(embedding_matrix, words)

    with tf.name_scope('encoder'):
        encoder_state = encoder_layer(word_vectors, lens)

        with tf.variable_scope('shared_vars', reuse=True):
            mu_style = ff_layer(encoder_state, name='mu_style')
            mu_content = ff_layer(encoder_state, name='mu_content')
            logvar_style = ff_layer(encoder_state, name='logvar_style')
            logvar_content = ff_layer(encoder_state, name='logvar_content')

        mu = tf.concat(1, [mu_style, mu_content])
        logvar = tf.concat(1, [logvar_style, logvar_content])

        z = sampling_layer(mu, logvar)

    # Decoder
    with tf.variable_scope('decoder'):
        outputs = decoder_layer(z, word_vectors, lens, eos_matrix)

    with tf.name_scope('loss_subtotal'):
        # Compute probabilities
        mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
        outputs_2D = tf.reshape(outputs, [-1, num_features])
        logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
        logits = tf.reshape(logits_2D, [BATCH_SIZE, -1, num_words])
        unmasked_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, words)
        softmax_loss = tf.mul(unmasked_softmax_loss, mask)
        batch_loss = tf.div(tf.reduce_sum(softmax_loss, reduction_indices=1), tf.cast(lens+1, tf.float32))
        mean_loss = tf.reduce_mean(batch_loss)

        KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1)
        KLD_word = tf.div(KLD, tf.cast(lens+1, tf.float32))
        mean_KLD = tf.reduce_mean(KLD_word)

    return mean_loss, mean_KLD, mu_style, mu_content, logvar_style, logvar_content, outputs


def initialize_shared_variables(lstm_size, latent_dim_size):
    with tf.variable_scope('shared_vars'):
        for scope_name in ['mu_style', 'mu_content', 'logvar_style', 'logvar_content']:
            init_ff_layer_vars(2 * lstm_size, int(latent_dim_size / 2), name=scope_name)
            # Note: There's a bit of magic going on here. These variables are initialized here
            # to be shared across multiple runs of hippopotamus, with the values being automatically
            # extracted as they are required


def initialize_embedding_variables(embedding):
    word_embedding = embedding.get_embedding_matrix()
    eos_embedding = embedding.get_eos_embedding()
    with tf.variable_scope("embedding"):
        tf.get_variable(
            "eos_matrix",
            initializer = tf.reshape(
                tf.tile(tf.constant(eos_embedding, dtype=tf.float32), [BATCH_SIZE]),
                [BATCH_SIZE, 1, embedding.get_num_features()],
            )
        )
        tf.get_variable("embedding_matrix", initializer=word_embedding)