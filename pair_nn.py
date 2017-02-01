import os
import time

import numpy as np
import tensorflow as tf

import batch
from embedding.embedder import word2vec
from embedding.w2vEmbedding import W2VEmbedding
from embedding.onehotEmbedding import OnehotEmbedding
from constants import BATCH_SIZE, CHECKPOINT_FILE, TB_LOGS_DIR, NUM_EPOCHS, STYLE_FRACTION
from nn_util import varec
from util import merge_dicts
from data.bible.training_data import read_bible

from nirv import pairs, common_pairs
CORPUS = pairs
COMMON_CORPUS = common_pairs

import argparse

parser = argparse.ArgumentParser(description='Embedding type (w2v or onehot)')
parser.add_argument('-emb', dest="embedding_type", default="w2v", action="store", type=str)

embedding_type = parser.parse_args().embedding_type

if (embedding_type == "w2v"):
    embedding = W2VEmbedding(word2vec)
else:
    embedding = OnehotEmbedding(word2vec)

with tf.name_scope('inputs'):
    sents = []
    lens = []
    for i in range(4):
        sents.append(tf.placeholder(tf.int32, [BATCH_SIZE, None], name='sentences'+str(i)))
        lens.append(tf.placeholder(tf.int32, [BATCH_SIZE], name='lengths'+str(i)))
    kl_weight = tf.placeholder(tf.float32, name='kl_weight')
    generation_state = tf.placeholder(tf.float32, [BATCH_SIZE, None], name='sentence')

with tf.variable_scope('shared') as scope:
    dicts = [varec(sents[0], lens[0], embedding, generation_state)]
    scope.reuse_variables()
    for i in range(1,4):
        dicts.append(varec(sents[i], lens[i], embedding, generation_state, summary=False))
    d = merge_dicts(dicts)

with tf.name_scope('loss_overall'):
    total_loss = 0
    for i in range(4):
        weighted_loss = d["loss"+str(i)] + kl_weight * d["kld"+str(i)]
        tf.scalar_summary("weighted loss "+str(i), weighted_loss)
        total_loss += weighted_loss

    total_nll = sum(d["loss"+str(i)] for i in range(4))
    total_kld = sum(d["kld"+str(i)] for i in range(4))
    content_penalty = tf.reduce_mean(tf.square(d["content0"]-d["content1"])) +\
        tf.reduce_mean(tf.square(d["content2"]-d["content3"]))
    style_penalty = tf.reduce_mean(tf.square(d["style0"]-d["style2"])) +\
        tf.reduce_mean(tf.square(d["style1"]-d["style3"])) -\
        tf.reduce_mean(tf.abs(d["style0"]-d["style1"])) -\
        tf.reduce_mean(tf.abs(d["style2"]-d["style3"]))
    z_penalty = content_penalty
    z_penalty = (1-STYLE_FRACTION)*content_penalty + STYLE_FRACTION*style_penalty
    total_loss += 20*z_penalty

    tf.scalar_summary('Total KLD', total_kld)
    tf.scalar_summary('Total NLL', total_nll)
    tf.scalar_summary('Content difference penalty', content_penalty)
    tf.scalar_summary('Style difference penalty', style_penalty)
    tf.scalar_summary('Z penalty', z_penalty)
    tf.scalar_summary('Total loss', total_loss)

train_step = tf.train.AdamOptimizer(1e-5).minimize(total_loss)

saver = tf.train.Saver()


timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
tensorboard_prefix = os.path.join(TB_LOGS_DIR, timestamp)

def _get_feed_dict(batches, klw):
    input_sentences = []
    input_lengths = []
    for bat in batches:
        sentence, length = embedding.word_indices(bat, eos=True)
        input_sentences.append(sentence)
        input_lengths.append(length)
    feed_dict={kl_weight:klw}
    for i in range(4):
        feed_dict[sents[i]] = input_sentences[i]
        feed_dict[lens[i]] = input_lengths[i]
    return feed_dict

def train():
    if not os.path.exists(tensorboard_prefix):
        print("Directory for checkpoints doesn't exist! Creating directory '%s'" % tensorboard_prefix)
        os.makedirs(tensorboard_prefix)
    else:
        print("Tensorboard logs will be saved to '%s'" % tensorboard_prefix)

    b = batch.Quads(CORPUS)
    epoch_length = b.num_training() // BATCH_SIZE
    summary_op = tf.merge_all_summaries()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        if os.path.isfile(CHECKPOINT_FILE):
            print("Restoring saved parameters")
            saver.restore(sess, CHECKPOINT_FILE)
        else:
            print("Initializing parameters")
            sess.run(tf.initialize_all_variables())
        summary_writer = tf.train.SummaryWriter(tensorboard_prefix, sess.graph)
        logging_iteration = 50
        output_iteration = 500
        start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            for i in range(epoch_length):
                batches = b.next_batch(BATCH_SIZE)
                global_step = i + epoch_length * epoch
                klw = .2
                _, los, _outputs, _mu_style, _mu_content, summary_str = sess.run((train_step, total_loss, d["outputs0"], d["style0"], d["content0"], summary_op), feed_dict=_get_feed_dict(batches,klw))
                if global_step%logging_iteration == 0:
                    summary_writer.add_summary(summary_str, global_step=global_step)
                    tpb = (time.time() - start_time) / logging_iteration
                    print("step {0}, training loss = {1} ({2} sec/batch)".format(global_step, los, tpb))
                    start_time = time.time()
                if global_step%output_iteration == 0:
                    mu = np.concatenate((_mu_style, _mu_content), axis=1)
                    gen_outputs = sess.run(d["generative_outputs0"], feed_dict={generation_state:mu})
                    gen_output = np.asarray(gen_outputs)[:,0,:]
                    print()
                    print('original      ' + batches[0][0])
                    print('with correct: ' + embedding.embedding_to_sentence(_outputs[0]))
                    print('using prev:   ' + embedding.embedding_to_sentence(gen_output))
                    print()
            # Validation loss
            batches = b.random_validation_batch(BATCH_SIZE)
            los = sess.run(total_loss, feed_dict=_get_feed_dict(batches,0))
            print()
            print("Epoch {0} validation loss: {1}".format(epoch, los))
            print()
            if epoch%5 == 4:
                saver.save(sess, CHECKPOINT_FILE)


def get_hidden(attribute_list):
    b = batch.Pairs(CORPUS)
    epoch_length = int(b.num_training()/BATCH_SIZE)
    hidden_representations1 = []
    hidden_representations2 = []
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILE)
        for i in range(epoch_length):
            batch1, batch2 = b.next_batch(BATCH_SIZE)
            sentences1, lengths1 = embedding.word_indices(batch1, eos=True)
            sentences2, lengths2 = embedding.word_indices(batch2, eos=True)
            attr_tuple1, attr_tuple2 = sess.run(
                    (tuple([d[attr+"0"] for attr in attribute_list]),
                     tuple([d[attr+"1"] for attr in attribute_list])),
                     feed_dict={
                         sents[0]:sentences1,
                         lens[0]:lengths1,
                         sents[1]:sentences2,
                         lens[1]:lengths2,
                         kl_weight:0})
            hidden_representations1.append(np.concatenate(attr_tuple1, axis=1))
            hidden_representations2.append(np.concatenate(attr_tuple2, axis=1))
    hidden_states1 = np.concatenate(hidden_representations1, axis=0)
    hidden_states2 = np.concatenate(hidden_representations2, axis=0)
    return hidden_states1, hidden_states2

def interpolate(k=5, use_z=True, use_content='avg'):
    b = batch.Pairs(COMMON_CORPUS)
    epoch_length = int(b.num_training()/BATCH_SIZE)
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_FILE)
        for i in range(epoch_length):
            batch1, batch2 = b.next_batch(BATCH_SIZE)
            sentences1, lengths1 = embedding.word_indices(batch1, eos=True)
            sentences2, lengths2 = embedding.word_indices(batch2, eos=True)
            pre = 'z_' if use_z else ''
            style1, style2, content1, content2 = sess.run((d[pre+'style0'], d[pre+'style1'], d[pre+'content0'], d[pre+'content1']), feed_dict={sents[0]:sentences1, lens[0]:lengths1, sents[1]:sentences2, lens[1]:lengths2, kl_weight:0})
            gen_output_words = [[] for _ in range(BATCH_SIZE)]
            content_avg = (content1 + content2) / 2
            for j in range(k+1):
                style_interp = (1 - j/k) * style1 + (j/k) * style2
                content_interp = (1 - j/k) * content1 + (j/k) * content2
                if use_content == 'interp':
                    content = content_interp
                elif use_content == 'avg':
                    content = content_avg
                elif use_content == '1':
                    content = content1
                elif use_content == '2':
                    content = content2
                else:
                    raise
                interp = np.concatenate((style_interp, content), axis=1)
                gen_outputs = sess.run(d['generative_outputs0'], feed_dict={generation_state:interp})
                for n in range(BATCH_SIZE):
                    gen_output = np.asarray(gen_outputs)[:,n,:]
                    gen_output_words[n].append(embedding.embedding_to_sentence(gen_output))
            for n in range(BATCH_SIZE):
                print('orig: %s' % batch1[n])
                for j in range(k+1):
                    v = j / k
                    print('%.02f: %s' % (v, gen_output_words[n][j]))
                print('orig: %s' % batch2[n])
                print()


if __name__ == "__main__":
    #interpolate()
    train()
#print(list(map(lambda x:x.name,tf.all_variables())))
