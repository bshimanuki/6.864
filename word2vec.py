import nltk
from gensim.models.word2vec import Word2Vec

WV_SIZE = 50
sents = nltk.corpus.brown.sents() + \
      nltk.corpus.gutenberg.sents('bible-kjv.txt')
sents = [[w.lower() for w in s] for s in sents]
word2vec = Word2Vec(sents,
        size=WV_SIZE)

# examples
print(word2vec['university'])
print(word2vec.most_similar('peter'))
