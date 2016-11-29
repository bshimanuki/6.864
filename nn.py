import tensorflow as tf
import embedding
import batch
import os
import time
import numpy as np

batch_size = 20
embedding_np = embedding.get_embedding_matrix()
num_words, num_features = embedding_np.shape
lstm_size = num_features
latent_dim = 2 * lstm_size
ckpt_file = 'output/temp.ckpt'
train_dir = 'train'

eos_embedding = embedding.get_eos_embedding()

with tf.name_scope("embedding"):
    eos_matrix = tf.reshape(tf.tile(tf.constant(
        eos_embedding, dtype=tf.float32),
        [batch_size]),
        [batch_size, 1, num_features])
    embedding_matrix = tf.Variable(tf.constant(0.0, shape=[num_words, num_features]),
                            trainable=False, name="embedding_matrix")
    embedding_placeholder = tf.placeholder(tf.float32, [num_words, num_features])
    embedding_init = embedding_matrix.assign(embedding_placeholder)

# Placeholder for the inputs in a given iteration
# NOTE: words is padded! Never add eos to the end of words!
with tf.name_scope("inputs"):
    words = tf.placeholder(tf.int32, [batch_size, None])
    lens = tf.placeholder(tf.int32, [batch_size])
    lens_plus_one = tf.add(lens, 1)

    word_vectors = tf.nn.embedding_lookup([embedding_matrix], words)
    eos_plus_words = tf.reverse(tf.slice(tf.reverse(tf.concat(
        1, [eos_matrix, word_vectors]),
        [False, True, False]),
        [0, 1, 0], [-1, -1, -1]),
        [False, True, False])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.0001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

with tf.name_scope('encoder'):
    with tf.variable_scope('encoder_lstm'):
        encoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)
        _, encoder_state = tf.nn.dynamic_rnn(encoder, word_vectors, sequence_length=lens, dtype=tf.float32)

    # Mu encoder
    with tf.name_scope('mu'):
        W_encoder_hidden_mu = weight_variable([2*lstm_size,latent_dim])
        b_encoder_hidden_mu = bias_variable([latent_dim])
        mu_encoder = tf.matmul(encoder_state, W_encoder_hidden_mu) + b_encoder_hidden_mu

    # Sigma encoder
    with tf.name_scope('sigma'):
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
    outputs, _ = tf.nn.dynamic_rnn(decoder, eos_plus_words, sequence_length=lens_plus_one, dtype=tf.float32)

with tf.name_scope('loss'):
    # Compute probabilities
    mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
    outputs_2D = tf.reshape(outputs, [-1, num_features])
    logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
    logits = tf.reshape(logits_2D, [batch_size, -1, num_words])
    unmasked_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, words)
    softmax_loss = tf.mul(unmasked_softmax_loss, mask)
    batch_loss = tf.div(tf.reduce_sum(softmax_loss, reduction_indices=1), tf.cast(lens_plus_one, tf.float32))

    KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)
    KLD_word = tf.div(KLD, tf.cast(lens_plus_one, tf.float32))
    mean_KLD = tf.reduce_mean(KLD_word)
    mean_loss = tf.reduce_mean(batch_loss)
    total_loss = mean_KLD + mean_loss
    tf.scalar_summary('KLD', mean_KLD)
    tf.scalar_summary('NLL', mean_loss)
    tf.scalar_summary('loss', total_loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(total_loss)

saver = tf.train.Saver()

def train():
    b = batch.Single()
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        if os.path.isfile(ckpt_file):
            print("Restoring saved parameters")
            saver.restore(sess, ckpt_file)
        else:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())
        sess.run(embedding_init, feed_dict={embedding_placeholder:embedding_np})
        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
        start_time = time.time()
        logging_iteration = 50
        for i in range(1, 200001):
            sentences, lengths = embedding.word_indices(b.next_batch(batch_size), eos=True)
            if i%logging_iteration == 0:
                _, los, summary_str = sess.run((train_step, total_loss, summary_op),
                        feed_dict={words:sentences, lens:lengths})
                tpb = (time.time() - start_time) / logging_iteration
                print("step {0}, loss = {1} ({2} sec/batch)".format(i, los, tpb))
                summary_writer.add_summary(summary_str, global_step=i)
                if i%1000 == 0:
                    saver.save(sess, ckpt_file)
                start_time = time.time()
            else:
                sess.run((train_step, total_loss), feed_dict={words:sentences, lens:lengths})

def test():
    b = batch.Single()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_file)
        bat = b.next_batch(batch_size)
        print(bat[0])
        for i in range(1):
            sentences, lengths = embedding.word_indices(bat, eos=True)
            _, output, los = sess.run((train_step, outputs, total_loss), feed_dict={words:sentences, lens:lengths})
        one_sentence = output[0]
        word_sequence = embedding.embedding_to_sentence(output[0])
        print(word_sequence)

train()
