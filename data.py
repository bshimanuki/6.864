import nltk
import numpy as np
from gensim.models import Word2Vec

wd = 50
word2vec = Word2Vec(map(lambda s: map(str.lower, s),
                            nltk.corpus.brown.sents() +
			    nltk.corpus.gutenberg.sents('bible-kjv.txt')),
                            size=wd)

def np_from_sentences(sentences):
    # TODO: Handle punctuation
    v = [[word2vec[word.lower()] for word in sentence.rstrip().split(" ")] for sentence in sentences]
    lens = np.array([len(sent) for sent in v])
    out = np.zeros((len(lens), max(lens), wd))
    for i, sent in enumerate(v):
        out[i, :lens[i]] = np.array(sent)
    return out, lens
