import itertools
import random
import pickle
import re

import nltk
import numpy as np

from word2vec import Word2Vec
from data.bible.book_titles import book_titles as bible_books

BOS = '<BOS>'
EOS = '<EOS>'

bible_path = 'data/bible/cache/by_book_filtered'

def process_sentence(sentence, bos=False, eos=False):
    # sentences are split by preprocessing in /data/bible/preprocess.py
    sentence = sentence.split()
    if eos:
        sentence = sentence + [EOS]
    if bos:
        sentence = [BOS] + sentence
    return sentence

def process_book(name, bos=False, eos=False):
    """
    Extracts all sentences from a specified book of the Bible.

    :param name:  Name of book of the Bible; data is expected to be stored as a *.pickle file.
    :param bos: Whether to append a BOS token to the sentences.
    :param eos: Whether to append an EOS token to the sentences.
    :return: List of all sentences from a specified book fo the Bible.
    """
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
    """
    Iterates over all sentences in the Bible.

    :param bos: Whether to append a BOS token to the sentences.
    :param eos: Whether to append an EOS token to the sentences.
    :return: Iterator iterating over Bible books in order (as specified in bible_books)
       with sentences randomly sorted within each Bible book.
    """
    def shuffled_book(book):
        text = process_book(book, bos=bos, eos=eos)
        random.shuffle(text)
        return text
    class Iterator:
        def __iter__(self):
            return itertools.chain(*map(shuffled_book, bible_books))
    return Iterator()

try:
    word2vec = Word2Vec.load('word2vec.pickle')
    print("Found and loaded word embedding.")
except FileNotFoundError:
    wd = 200
    print("Generating word embeddings from scratch.")
    word2vec = Word2Vec(process_bible(eos=True),
            size=wd)
    #word2vec.normalize()
    word2vec.save('word2vec.pickle')

