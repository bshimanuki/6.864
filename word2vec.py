import itertools

import gensim
import nltk
import numpy as np

class Word2Vec(gensim.models.word2vec.Word2Vec):
    def __init__(self, *args, **kwargs):
        kwargs['null_word'] = True
        super().__init__(*args, **kwargs)
        self.null_word = '\0' # from internal gensim model

    # We normalize based on word frequencies
    def normalize(self):
        # null_word does not have freq counts, so set one before normalization
        self.vocab[self.null_word].count = self.min_count
        self.init_sims(replace=True)
        counts = np.zeros(len(self.vocab))
        for word in self.vocab.values():
            counts[word.index] = word.count
        counts = counts.reshape((-1,1))
        # log because softmax will recover the frequency proportions
        self.syn0 *= np.log(counts)

    def one_hot(self, words, sent_length=None):
        if sent_length == None:
            sent_length = len(words)
        ret = np.zeros((sent_length, self.syn0.shape[0]))
        for i, word in enumerate(words):
            if word in self.vocab:
                ret[i][self.vocab[word].index] = 1
            else:
                ret[i][self.vocab[self.null_word].index] = 1
        return ret

    def words_to_indices(self, words, sent_length=None):
        if sent_length == None:
            sent_length = len(words)
        ret = np.zeros(sent_length)
        for i, word in enumerate(words):
            if word in self.vocab:
                ret[i] = self.vocab[word].index
            else:
                ret[i] = self.vocab[self.null_word].index
        return ret

    def embedding_matrix(self):
        return self.syn0

    def get_vocab(self):
        return list(self.vocab.keys())

    # Calculate loss given word vectors and target words.
    # vectors is an array or tensor with size (n,d) of n word vectors.
    # words is a list of n words to compute the loss against.
    # Returns a tensor with size (n,).
    """
    def loss(self, vectors, words):
        dist = tf.matmul(self.syn0, vectors, transpose_b=True)
        dist = tf.transpose(dist)
        words = [self.vocab[word].index
                if word in self.vocab
                else self.vocab[self.null_word].index
                for word in words]
        return tf.nn.sparse_softmax_cross_entropy_with_logits(dist, words)
    """
