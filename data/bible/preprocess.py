# -*- coding: utf-8 -*-
"""
Sanitize data. Removes editor notes. Splits into tokens.
Transforms pickle dict keys to int chapter and verse and strips parentheses from versions.
"""

import pickle
import re
import unicodedata

import nltk

from book_titles import book_titles

trans = str.maketrans('{}‘’“”\x0a', '[]\'\'"" ')
def parse_sentence(sentence):
    # remove [+ notes]
    sentence = re.sub(r'\[\+.*?\]', r'', sentence)
    # remove <note:>
    sentence = re.sub(r'<note:.*?>', r'', sentence)
    # remove {Gr. notes}
    sentence = re.sub(r'{Gr\..*?}', r'', sentence)
    # remove * (markdown bold)
    sentence = re.sub(r'\*', r'', sentence)
    # remove <em> tags
    sentence = re.sub(r'</?em>', r'', sentence)
    # convert to canonical symbols
    sentence = sentence.translate(trans)
    sentence = re.sub(r'[–—]+', r'--', sentence)
    # remove accents
    sentence = ''.join([c for c in unicodedata.normalize('NFKD', sentence)
        if not unicodedata.combining(c)])
    # make lowercase
    sentence = sentence.lower()
    # split sentence into tokens
    sentence = nltk.tokenize.word_tokenize(sentence)
    return ' '.join(sentence)

def convert_book(book_title, remove=[]):
    remove = set(map(''.join, map(lambda x:filter(str.isalpha, x), remove)))
    with open('cache/by_book/%s.pickle' % book_title, 'rb') as f:
        book = pickle.load(f)
    output = {}
    for chapter, verses in sorted(book.items()):
        chapter = int(chapter)
        output[chapter] = {}
        for verse, translations in sorted(verses.items()):
            verse = int(verse)
            output[chapter][verse] = {}
            for trans, sent in sorted(translations.items()):
                trans = ''.join(filter(str.isalpha, trans))
                if trans not in remove:
                    if sent is not None:
                        sent = parse_sentence(sent)
                    output[chapter][verse][trans] = sent
    return output

def convert_bible(remove=[]):
    for book_title in book_titles:
        book = convert_book(book_title, remove=remove)
        pickle.dump(book, open("cache/by_book_filtered/{0}.pickle".format(book_title), "wb"))

if __name__ == '__main__':
    """

    To see the text for a sample chapter of a translation, navigate to (for example, for the HNV, Genesis 1)
        http://www.biblestudytools.com/hnv/genesis/1.html

    Translations removed by default:
        (OJB) Way too many Jewish terms
        (TYN) Written in old English; likely to mess up our word2vec. Also missing (at least) a large chunk of the
            Old Testament.
        (SBLG) Written in Greek (!). Only contains the New Testament.


    Additional candidates for removal:
        (HNV) Has some weird names for words (e.g. in Genesis, earth is referred to as Eretz)
        (WNT) Only contains the New Testament.
        (LXX) Only contains the Old Testament.
    """
    # remove_unused_translations(unused_translations=['(OJB)', '(TYN)', '(SBLG)']):
    convert_bible(remove=['(OJB)', '(TYN)', '(SBLG)', '(HNV)', '(WNT)', '(LXX)'])
