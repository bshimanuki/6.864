import tensorflow as tf

from constants import BATCH_SIZE, MAX_GENERATION_SIZE
import embedder


def weight_variable(shape, name, summary):
    assert name is not None
    full_name = '%s/weight' % (name)
    weight = tf.get_variable(full_name, initializer=tf.truncated_normal(shape, stddev=0.0001))
    if summary:
        tf.histogram_summary(full_name, weight)
    return weight


def bias_variable(shape, name, summary):
    assert name is not None
    full_name = '%s/bias' % (name)
    bias = tf.get_variable(full_name, initializer=tf.zeros(shape=shape))
    if summary:
        tf.histogram_summary(full_name, bias)
    return bias


def ff_layer(input_layer, input_depth, output_depth, name=None, summary=True):
    output = tf.matmul(input_layer, weight_variable([input_depth, output_depth], name, summary)) +\
        bias_variable([output_depth], name, summary)
    return output


def encoder_layer(word_vectors, lens, lstm_size):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)
    _, encoder_state = tf.nn.dynamic_rnn(lstm_cell, word_vectors, sequence_length=lens, dtype=tf.float32)
    return encoder_state


def sampling_layer(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')

    # Sample latent variable
    std_encoder = tf.exp(0.5 * logvar)
    z = mu + tf.mul(std_encoder, epsilon)
    return z


def decoder_layer(z, word_vectors, lens, eos_matrix, lstm_size):
    eos_plus_words = tf.reverse(tf.slice(tf.reverse(tf.concat(
        1, [eos_matrix, word_vectors]),
        [False, True, False]),
        [0, 1, 0], [-1, -1, -1]),
        [False, True, False])
    decoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)
    outputs, _ = tf.nn.dynamic_rnn(decoder, eos_plus_words, sequence_length=lens+1, initial_state=z, dtype=tf.float32, scope='RNN')
    return outputs, decoder


def generative_decoder_layer(cell, initial_state, embedding_matrix, eos_matrix):
    """Adapted from definition of tf.nn.seq2seq.rnn_decoder."""
    with tf.variable_scope('RNN', reuse=True):
        state = initial_state
        outputs = []
        output_words = []
        inp = tf.reshape(eos_matrix, [BATCH_SIZE, -1])
        for i in range(MAX_GENERATION_SIZE):
            output, state = cell(inp, state)

            logits = tf.matmul(output, embedding_matrix, transpose_b=True)
            words = tf.argmax(logits, dimension=1)
            inp = tf.nn.embedding_lookup(embedding_matrix, words)

            outputs.append(output)
            output_words.append(words)
        return outputs, output_words


def varec(words_placeholder, lens, embedding, style_fraction, generation_state, summary=True):
    num_features = embedding.get_num_features()
    num_words = embedding.get_vocabulary_size()
    lstm_size = num_features
    content_size = int(2*lstm_size * (1-style_fraction))
    style_size = 2*lstm_size - content_size

    with tf.name_scope('embedding'):
        eos_matrix = tf_eos_matrix(embedding)
        embedding_matrix = tf_embedding_matrix(embedding)

    word_vectors = tf.nn.embedding_lookup(embedding_matrix, words_placeholder)

    with tf.variable_scope('encoder'):
        with tf.name_scope('encoder_rnn'):
            encoder_state = encoder_layer(word_vectors, lens, lstm_size)

        mu_style = ff_layer(encoder_state, 2*lstm_size, style_size, name='mu_style', summary=summary)
        mu_content = ff_layer(encoder_state, 2*lstm_size, content_size, name='mu_content', summary=summary)
        logvar_style = ff_layer(encoder_state, 2*lstm_size, style_size, name='logvar_style', summary=summary)
        logvar_content = ff_layer(encoder_state, 2*lstm_size, content_size, name='logvar_content', summary=summary)

        mu = tf.concat(1, [mu_style, mu_content])
        logvar = tf.concat(1, [logvar_style, logvar_content])

        z = sampling_layer(mu, logvar)

    # Decoder
    with tf.variable_scope('decoder'):
        outputs, decoder_cell = decoder_layer(z, word_vectors, lens, eos_matrix, lstm_size)

        generative_outputs, generative_output_words = generative_decoder_layer(decoder_cell, generation_state, embedding_matrix, eos_matrix)

    with tf.name_scope('loss_subtotal'):
        # Compute probabilities
        mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
        outputs_2D = tf.reshape(outputs, [-1, num_features])
        logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
        logits = tf.reshape(logits_2D, [BATCH_SIZE, -1, num_words])
        unmasked_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, words_placeholder)
        softmax_loss = tf.mul(unmasked_softmax_loss, mask)
        batch_loss = tf.div(tf.reduce_sum(softmax_loss, reduction_indices=1), tf.cast(lens+1, tf.float32))
        mean_loss = tf.reduce_mean(batch_loss)

        KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1)
        KLD_word = tf.div(KLD, tf.cast(lens+1, tf.float32))
        mean_KLD = tf.reduce_mean(KLD_word)

    return mean_loss, mean_KLD, mu_style, mu_content, outputs, generative_outputs, z

def tf_eos_matrix(embedding):
    return tf.reshape(
        tf.tile(tf.constant(embedding.get_eos_embedding(), dtype=tf.float32), [BATCH_SIZE]),
        [BATCH_SIZE, 1, embedding.get_num_features()])

def tf_embedding_matrix(embedding):
    matr_np = embedding.get_embedding_matrix()
    return tf.get_variable("embedding_matrix", shape=matr_np.shape, initializer=tf.constant_initializer(matr_np), trainable=False)
