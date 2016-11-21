__author__ = 'vince_000'
import pickle
import os

"""
Reads cached information to generate training data.
"""

def get_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)

def get_corresponding_sentences_in_book(book_output, trans1, trans2):
    """
    Gets corresponding sentences for a given book of the bible for two specified translations.

    :param book_output:
    :param trans1: First translation. e.g. '(WYC)'
    :param trans2: Second translation e.g. '(WEB)'
    :return: List of tuples of the verse text, with the first translation first.
    """
    corresponding_sentences = list()
    for chapter_output in book_output.values():
        for verse_output in chapter_output.values():
            if trans1 in verse_output and trans2 in verse_output:
                # TODO: consider whether we want to throw away translations where the verse is empty.
                corresponding_sentences.append((verse_output[trans1], verse_output[trans2]))
    return corresponding_sentences

def get_corresponding_sentences_in_bible(trans1, trans2):
    """
    Get corresponding sentences in all books of the bible for all translations.
    :param trans1:
    :param trans2:
    :return:
    """
    book_titles = pickle.load(open(get_path('cache/book_titles.pickle'), 'rb'))
    corresponding_sentences = []
    for book_title in book_titles:
        book_output = pickle.load(open(get_path("cache/by_book/{0}.pickle".format(book_title)), "rb"))
        pairs = get_corresponding_sentences_in_book(book_output, trans1, trans2)
        corresponding_sentences.extend(pairs)
    return corresponding_sentences
