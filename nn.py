import os
import time

import tensorflow as tf

import batch
import embedding
from constants import CORPUS, BATCH_SIZE, KL_PARAM, KL_TRANSLATE, CHECKPOINT_FILE, TRAIN_DIR
from nn_util import ff_layer, ff_layer_vars, encoder_layer, sampling_layer, decoder_layer
from util import sigmoid

embedding_np = embedding.get_embedding_matrix()
num_words, num_features = embedding_np.shape
lstm_size = num_features
latent_dim_size = 2 * lstm_size
half_latent_dim_size = int(latent_dim_size/2) # TODO: this is a badbadbad hack

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

    word_vectors = tf.nn.embedding_lookup([embedding_matrix], words)

with tf.name_scope('encoder'):
    encoder_state = encoder_layer(word_vectors, lens)

    (w_mu_style, b_mu_style) = ff_layer_vars(2*lstm_size, half_latent_dim_size, name='mu_style')
    (w_mu_content, b_mu_content) = ff_layer_vars(2*lstm_size, half_latent_dim_size, name='mu_content')
    (w_logvar_style, b_logvar_style) = ff_layer_vars(2*lstm_size, half_latent_dim_size, name='logvar_style')
    (w_logvar_content, b_logvar_content) = ff_layer_vars(2*lstm_size, half_latent_dim_size, name='logvar_content')

    mu_style = ff_layer(encoder_state, w_mu_style, b_mu_style, name='mu_style')
    mu_content = ff_layer(encoder_state, w_mu_content, b_mu_content, name='mu_content')
    logvar_style = ff_layer(encoder_state, w_logvar_style, b_logvar_style, name='logvar_style')
    logvar_content = ff_layer(encoder_state, w_logvar_content, b_logvar_content, name='logvar_content')
    
    mu = tf.concat(1, [mu_style, mu_content])
    logvar = tf.concat(1, [logvar_style, logvar_content])

    z = sampling_layer(mu, logvar)

# Decoder
with tf.variable_scope('decoder'):
    outputs = decoder_layer(z, word_vectors, lens, eos_matrix)

with tf.name_scope('loss'):
    # Compute probabilities
    mask = tf.sign(tf.reduce_max(tf.abs(outputs), reduction_indices=2))
    outputs_2D = tf.reshape(outputs, [-1, num_features])
    logits_2D = tf.matmul(outputs_2D, embedding_matrix, transpose_b=True)
    logits = tf.reshape(logits_2D, [BATCH_SIZE, -1, num_words])
    unmasked_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, words)
    softmax_loss = tf.mul(unmasked_softmax_loss, mask)
    batch_loss = tf.div(tf.reduce_sum(softmax_loss, reduction_indices=1), tf.cast(lens+1, tf.float32))
    mean_loss = tf.reduce_mean(batch_loss)
    
    kl_weight = tf.placeholder(tf.float32)
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1)
    KLD_word = tf.div(KLD, tf.cast(lens+1, tf.float32))
    mean_KLD = tf.reduce_mean(KLD_word)
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
