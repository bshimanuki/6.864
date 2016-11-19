import itertools
import random
import pickle
import re

import nltk
import numpy as np

from word2vec import Word2Vec

OUTPUT = 'output'

bible_path = 'data/bible/cache'
bible_books = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Songs', 'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi', 'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude', 'Revelation']

def process_sentence(sentence):
    sentence = re.sub(r'(\W+)', r' \1 ', sentence)
    return [w.lower() for w in sentence.split()]

def process_book(name):
    sents = []
    with open('%s/%s.pickle' % (bible_path, name), 'rb') as f:
        book = pickle.load(f)
        for _, chapter in sorted(book.items()):
            for _, verse in sorted(chapter.items()):
                for _, trans in sorted(verse.items()):
                    if trans is not None:
                        sents.append(process_sentence(trans))
    return sents

def process_bible():
    def shuffled_book(book):
        text = process_book(book)
        random.shuffle(text)
        return text
    class Iterator:
        def __iter__(self):
            return itertools.chain(*map(shuffled_book, bible_books))
    return Iterator()

def np_from_sentences(sentences):
    # TODO: Handle punctuation
    v = [[word2vec[word.lower()] for word in sentence.rstrip().split(" ")] for sentence in sentences]
    lens = np.array([len(sent) for sent in v])
    out = np.zeros((len(lens), max(lens), wd))
    for i, sent in enumerate(v):
        out[i, :lens[i]] = np.array(sent)
    return out, lens

wd = 50
try:
    word2vec = Word2Vec.load(OUTPUT + '/word2vec.pickle')
except FileNotFoundError:
    word2vec = Word2Vec(process_bible(),
            size=wd)
    word2vec.normalize()
    word2vec.save(OUTPUT + '/word2vec.pickle')

