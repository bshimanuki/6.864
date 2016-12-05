import os
import time

import tensorflow as tf

import batch
from embedder import word2vec
from w2vEmbedding import W2VEmbedding
from constants import CORPUS, BATCH_SIZE, KL_PARAM, KL_TRANSLATE, CHECKPOINT_FILE, TB_LOGS_DIR
from nn_util import varec, initialize_shared_variables, initialize_embedding_variables
from util import sigmoid

embedding = W2VEmbedding(word2vec)
style_fraction = .01

kl_sigmoid = sigmoid(KL_PARAM, KL_TRANSLATE)

with tf.name_scope('inputs'):
    # Placeholder for the inputs in a given iteration
    # NOTE: words is padded! Never add eos to the end of words!
    words = tf.placeholder(tf.int32, [BATCH_SIZE, None], name = 'words')
    lens = tf.placeholder(tf.int32, [BATCH_SIZE], name = 'lengths')
    kl_weight = tf.placeholder(tf.float32, name='kl_weight')

with tf.variable_scope('shared'):
    (mean_loss, mean_KLD, mu_style, mu_content, logvar_style, logvar_content, outputs) = varec(words, lens, embedding, style_fraction)

with tf.name_scope('loss_overall'):
    total_loss = kl_weight*mean_KLD + mean_loss

tf.scalar_summary('KLD', mean_KLD)
tf.scalar_summary('NLL', mean_loss)
tf.scalar_summary('loss', total_loss)

train_step = tf.train.AdamOptimizer(0.0001).minimize(total_loss)

saver = tf.train.Saver()


timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
tensorboard_prefix = os.path.join(TB_LOGS_DIR, timestamp)

if not os.path.exists(tensorboard_prefix):
    print("Directory for checkpoints doesn't exist! Creating directory '%s'" % tensorboard_prefix)
    os.makedirs(tensorboard_prefix)
else:
    print("Tensorboard logs will be saved to '%s'" % tensorboard_prefix)


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
        summary_writer = tf.train.SummaryWriter(tensorboard_prefix, sess.graph)
        start_time = time.time()
        logging_iteration = 50
        for i in range(1, 200001):
            sentences, lengths = embedding.word_indices(b.next_batch(BATCH_SIZE), eos=True)
            _, los, summary_str = sess.run((train_step, total_loss, summary_op),
                    feed_dict={words:sentences, lens:lengths, kl_weight:kl_sigmoid(i)})
            summary_writer.add_summary(summary_str, global_step=i)
            if i%logging_iteration == 0:
                tpb = (time.time() - start_time) / logging_iteration
                print("step {0}, loss = {1} ({2} sec/batch)".format(i, los, tpb))
                if i%1000 == 0:
                    saver.save(sess, CHECKPOINT_FILE)
                start_time = time.time()

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
