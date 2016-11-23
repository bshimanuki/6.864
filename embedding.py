import itertools
import random
import pickle
import re

import nltk
import numpy as np

from word2vec import Word2Vec
from data.bible.book_titles import book_titles as bible_books

OUTPUT = 'output'
BOS = '<BOS>'
EOS = '<EOS>'

bible_path = 'data/bible/cache/by_book_filtered'

def process_sentence(sentence, bos=False, eos=False):
    # setences are split by preprocessing in /data/bible/preprocess.py
    sentence = sentence.split()
    if eos:
        sentence = sentence + [EOS]
    if bos:
        sentence = [BOS] + sentence
    return sentence

def process_book(name, bos=False, eos=False):
    sents = []
    with open('%s/%s.pickle' % (bible_path, name), 'rb') as f:
        book = pickle.load(f)
        for _, chapter in sorted(book.items()):
            for _, verse in sorted(chapter.items()):
                for _, trans in sorted(verse.items()):
                    if trans is not None:
                        sents.append(process_sentence(trans, bos=bos, eos=eos))
    return sents

def process_bible(bos=False, eos=False):
    def shuffled_book(book):
        text = process_book(book, bos=bos, eos=eos)
        random.shuffle(text)
        return text
    class Iterator:
        def __iter__(self):
            return itertools.chain(*map(shuffled_book, bible_books))
    return Iterator()

def word_indices(sentences, eos=False):
    if eos:
        sents = [process_sentence(sent, eos=True) for sent in sentences]
        lens = np.array([len(sent) - 1 for sent in sents])
        max_len = max(lens) + 1
    else:
        sents = [process_sentence(sent) for sent in sentences]
        lens = np.array([len(sent) for sent in sents])
        max_len = max(lens)
    v = np.stack([word2vec.words_to_indices(sent, sent_length=max_len) for sent in sents], axis=0)
    return v, lens

def get_embedding_matrix():
    return word2vec.embedding_matrix()

def get_eos_embedding():
    return get_embedding_matrix()[word2vec.words_to_indices(['<EOS>'])[0]]

def embedding_to_sentence(embeddings):
    vocab = word2vec.index_to_word_map()
    word_probs = np.matmul(embeddings, np.transpose(get_embedding_matrix()))
    num_words_sentence, num_words_vocab = word_probs.shape
    word_sequence = [vocab[np.argmax(word_probs[i])] for i in range(num_words_sentence)]
    return ' '.join(word_sequence)

wd = 200
try:
    word2vec = Word2Vec.load(OUTPUT + '/word2vec.pickle')
    print("Found and loaded word embedding.")
except FileNotFoundError:
    print("Generating word embeddings from scratch.")
    word2vec = Word2Vec(process_bible(eos=True),
            size=wd)
    #word2vec.normalize()
    word2vec.save(OUTPUT + '/word2vec.pickle')

