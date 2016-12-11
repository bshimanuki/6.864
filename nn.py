import os
import time

import numpy as np
import tensorflow as tf

import batch
from embedder import word2vec
from w2vEmbedding import W2VEmbedding
from onehotEmbedding import OnehotEmbedding
from constants import BATCH_SIZE, KL_PARAM, KL_TRANSLATE, CHECKPOINT_FILE, TB_LOGS_DIR, NUM_EPOCHS
from nn_util import varec
from util import sigmoid

from nirv import sentences
CORPUS = sentences

import argparse

parser = argparse.ArgumentParser(description='Embedding type (w2v or onehot)')
parser.add_argument('-emb', dest="embedding_type", default="w2v", action="store", type=str)

embedding_type = parser.parse_args().embedding_type

if (embedding_type == "w2v"):
    embedding = W2VEmbedding(word2vec)
else:
    embedding = OnehotEmbedding(word2vec)

style_fraction = .01

kl_sigmoid = sigmoid(KL_PARAM, KL_TRANSLATE)

with tf.name_scope('inputs'):
    # Placeholder for the inputs in a given iteration
    # NOTE: words is padded! Never add eos to the end of words!
    words = tf.placeholder(tf.int32, [BATCH_SIZE, None], name = 'words')
    lens = tf.placeholder(tf.int32, [BATCH_SIZE], name = 'lengths')
    kl_weight = tf.placeholder(tf.float32, name='kl_weight')
    generation_state = tf.placeholder(tf.float32, [BATCH_SIZE, None], name='sentence')

with tf.variable_scope('shared'):
    (mean_loss, mean_KLD, mu_style, mu_content, outputs, generative_outputs, _) = varec(words, lens, embedding, style_fraction, generation_state)

with tf.name_scope('loss_overall'):
    total_loss = mean_loss + kl_weight*mean_KLD

tf.scalar_summary('KL weight', kl_weight)
tf.scalar_summary('KLD', mean_KLD)
tf.scalar_summary('NLL', mean_loss)
tf.scalar_summary('loss', total_loss)

train_step = tf.train.AdamOptimizer(0.0001).minimize(total_loss)

saver = tf.train.Saver()


timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
tensorboard_prefix = os.path.join(TB_LOGS_DIR, timestamp)


def train():
    if not os.path.exists(tensorboard_prefix):
        print("Directory for checkpoints doesn't exist! Creating directory '%s'" % tensorboard_prefix)
        os.makedirs(tensorboard_prefix)
    else:
        print("Tensorboard logs will be saved to '%s'" % tensorboard_prefix)

    b = batch.Single(CORPUS)
    epoch_length = b.num_training() // BATCH_SIZE
    summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
        if os.path.isfile(CHECKPOINT_FILE):
            print("Restoring saved parameters")
            saver.restore(sess, CHECKPOINT_FILE)
        else:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(tensorboard_prefix, sess.graph)
        logging_iteration = 50
        output_iteration = 500
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            for i in range(epoch_length):
                next_batch = b.next_batch(BATCH_SIZE)
                sentences, lengths = embedding.word_indices(next_batch, eos=True)
                global_step = epoch * epoch_length + i
                klw = kl_sigmoid(epoch) / 5
                _, los, _outputs, _mu_style, _mu_content, summary_str = sess.run(
                        (train_step, total_loss, outputs, mu_style, mu_content, summary_op),
                        feed_dict={words:sentences, lens:lengths, kl_weight:klw})
                if global_step%logging_iteration == 0:
                    summary_writer.add_summary(summary_str, global_step=global_step)
                    tpb = (time.time() - start_time) / logging_iteration
                    print("step {0}, training loss = {1} ({2} sec/batch)".format(global_step, los, tpb))
                    start_time = time.time()
                if global_step%output_iteration == 0:
                    mu = np.concatenate((_mu_style, _mu_content), axis=1)
                    gen_outputs = sess.run(generative_outputs, feed_dict={generation_state:mu})
                    gen_output = np.asarray(gen_outputs)[:,0,:]
                    print()
                    print('original:     ' + next_batch[0])
                    print('with correct: ' + embedding.embedding_to_sentence(_outputs[0]))
                    print('using prev:   ' + embedding.embedding_to_sentence(gen_output))
                    print()
            # Validation loss
            sentences, lengths = embedding.word_indices(b.random_validation_batch(BATCH_SIZE), eos=True)
            los = sess.run(total_loss, feed_dict={words:sentences, lens:lengths, kl_weight: 0})
            print()
            print("Epoch {0} validation loss: {1}".format(epoch, los))
            print()
            if epoch%5 == 4:
                saver.save(sess, CHECKPOINT_FILE)


def test():
    b = batch.Single(CORPUS)
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILE)
        bat = b.next_batch(BATCH_SIZE)
        print(bat[0])
        for i in range(1):
            sentences, lengths = embedding.word_indices(bat, eos=True)
            output, los = sess.run((outputs, total_loss), feed_dict={words:sentences, lens:lengths, kl_weight:0})
        one_sentence = output[0]
        word_sequence = embedding.embedding_to_sentence(output[0])
        print(word_sequence)

train()
