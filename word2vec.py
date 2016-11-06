#/bin/python

import nltk
from gensim.models import Word2Vec

WV_SIZE = 50
word2vec = Word2Vec(
		map(lambda s:map(unicode.lower, s),
			nltk.corpus.brown.sents() +
			nltk.corpus.gutenberg.sents('bible-kjv.txt')),
		size=WV_SIZE)

# examples
print(word2vec['university'])
print(word2vec.most_similar('peter'))
