import os
import time

import tensorflow as tf

import batch
import embedding
from constants import CORPUS, BATCH_SIZE, KL_PARAM, KL_TRANSLATE, CHECKPOINT_FILE, TRAIN_DIR
from nn_util import ff_layer
from util import sigmoid

embedding_np = embedding.get_embedding_matrix()
num_words, num_features = embedding_np.shape
lstm_size = num_features
latent_dim = 2 * lstm_size

eos_embedding = embedding.get_eos_embedding()

kl_sigmoid = sigmoid(KL_PARAM, KL_TRANSLATE)

with tf.name_scope("embedding"):
    eos_matrix = tf.reshape(tf.tile(tf.constant(
        eos_embedding, dtype=tf.float32),
        [BATCH_SIZE]),
        [BATCH_SIZE, 1, num_features])
    embedding_matrix = tf.Variable(embedding_np, name="embedding_matrix")

# Placeholder for the inputs in a given iteration
# NOTE: words is padded! Never add eos to the end of words!
with tf.name_scope("inputs"):
    words = tf.placeholder(tf.int32, [BATCH_SIZE, None])
    lens = tf.placeholder(tf.int32, [BATCH_SIZE])
    lens_plus_one = tf.add(lens, 1)
    kl_weight = tf.placeholder(tf.float32)

    word_vectors = tf.nn.embedding_lookup([embedding_matrix], words)
    eos_plus_words = tf.reverse(tf.slice(tf.reverse(tf.concat(
        1, [eos_matrix, word_vectors]),
        [False, True, False]),
        [0, 1, 0], [-1, -1, -1]),
        [False, True, False])

with tf.name_scope('encoder'):
    with tf.variable_scope('encoder_lstm'):
        encoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)
        _, encoder_state = tf.nn.dynamic_rnn(encoder, word_vectors, sequence_length=lens, dtype=tf.float32)

    mu = ff_layer(encoder_state, latent_dim, name='mu')
    logvar = ff_layer(encoder_state, latent_dim, name='sigma')

    # Sample epsilon
    epsilon = tf.random_normal(tf.shape(logvar), name='epsilon')

    # Sample latent variable
    std_encoder = tf.exp(0.5 * logvar)
    z = mu + tf.mul(std_encoder, epsilon)

# Decoder
with tf.variable_scope('decoder'):
    decoder = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=False)
    outputs, _ = tf.nn.dynamic_rnn(decoder, eos_plus_words, sequence_length=lens_plus_one, initial_state=z, dtype=tf.float32)

with tf.name_scope('loss'):
    # Compute probabilities
    mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
    outputs_2D = tf.reshape(outputs, [-1, num_features])
    logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
    logits = tf.reshape(logits_2D, [BATCH_SIZE, -1, num_words])
    unmasked_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, words)
    softmax_loss = tf.mul(unmasked_softmax_loss, mask)
    batch_loss = tf.div(tf.reduce_sum(softmax_loss, reduction_indices=1), tf.cast(lens_plus_one, tf.float32))

    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1)
    KLD_word = tf.div(KLD, tf.cast(lens_plus_one, tf.float32))
    mean_KLD = tf.reduce_mean(KLD_word)
    mean_loss = tf.reduce_mean(batch_loss)
    total_loss = kl_weight*mean_KLD + mean_loss
    tf.scalar_summary('KLD', mean_KLD)
    tf.scalar_summary('NLL', mean_loss)
    tf.scalar_summary('loss', total_loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(total_loss)

saver = tf.train.Saver()

def train():
    b = batch.Single(CORPUS)
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        if os.path.isfile(CHECKPOINT_FILE):
            print("Restoring saved parameters")
            saver.restore(sess, CHECKPOINT_FILE)
        else:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(TRAIN_DIR, sess.graph)
        start_time = time.time()
        logging_iteration = 50
        for i in range(1, 200001):
            sentences, lengths = embedding.word_indices(b.next_batch(BATCH_SIZE), eos=True)
            if i%logging_iteration == 0:
                _, los, summary_str = sess.run((train_step, total_loss, summary_op),
                        feed_dict={words:sentences, lens:lengths, kl_weight:kl_sigmoid(i)})
                tpb = (time.time() - start_time) / logging_iteration
                print("step {0}, loss = {1} ({2} sec/batch)".format(i, los, tpb))
                summary_writer.add_summary(summary_str, global_step=i)
                if i%1000 == 0:
                    saver.save(sess, CHECKPOINT_FILE)
                start_time = time.time()
            else:
                sess.run((train_step, total_loss), feed_dict={words:sentences, lens:lengths, kl_weight:kl_sigmoid(i)})

def test():
    b = batch.Single(CORPUS)
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILE)
        bat = b.next_batch(BATCH_SIZE)
        print(bat[0])
        for i in range(1):
            sentences, lengths = embedding.word_indices(bat, eos=True)
            _, output, los = sess.run((train_step, outputs, total_loss), feed_dict={words:sentences, lens:lengths, kl_weight:kl_sigmoid(i)})
        one_sentence = output[0]
        word_sequence = embedding.embedding_to_sentence(output[0])
        print(word_sequence)

train()
