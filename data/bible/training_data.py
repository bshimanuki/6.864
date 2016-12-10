__author__ = 'vince_000'
import pickle
import os

from . import book_titles

"""
Reads cached information to generate training data.
"""

def get_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)

def read_bible(books=[], chapters=[], verses=[], translations=[],
        flatten_books=True, flatten_chapters=True, flatten_verses=True,
        flatten_translations=False):
    """Reads the bible verses and translations from the pickles, returning a list (if flatten_* are all True) or a recursive dict with the flatten_*=False as keys.

    Use this when you want lists of sentences. Use the get_corresponding_sentences_in_* when you want parallel sentences.

    :param books: a list of books to get, or the empty list for all books
    :param chapters: a list of chapters to get, or the empty list for all chapters
    :param verses: a list of verses to get, or the empty list for all verses
    :param translations: a list of translations to get, or the empty list for all translations
    :param flatten_*: flatten the level of the dict corresponding to *
    """
    books = set(books)
    chapters = set(chapters)
    verses = set(verses)
    translations = set(translations)

    output = {}
    for book in book_titles:
        if books and book not in books:
            continue
        if flatten_books:
            output_b = output
        else:
            output[book] = {}
            output_b = output[book]
        with open(get_path('cache/by_book_filtered/%s.pickle' % book), 'rb') as f:
            book_dict = pickle.load(f)
        for chapter, verses_dict in sorted(book_dict.items()):
            if chapters and chapter not in chapters:
                continue
            if flatten_chapters:
                output_c = output_b
            else:
                output_b[chapter] = {}
                output_c = output_b[chapter]
            for verse, translations_dict in sorted(verses_dict.items()):
                if verses and verse not in verses:
                    continue
                if flatten_verses:
                    output_v = output_c
                else:
                    output_c[verse] = {}
                    output_v = output_c[verse]
                for translation, sent in sorted(translations_dict.items()):
                    if translations and translation not in translations:
                        continue
                    if flatten_translations:
                        key = None
                    else:
                        key = translation
                    if key not in output_v:
                        output_v[key] = []
                    output_v[key].append(sent)
                if None in output_v:
                    output_c[verse] = output_v[None]
            if None in output_c:
                output_b[chapter] = output_c[None]
        if None in output_b:
            output[book] = output_b[None]
    if None in output:
        output = output[None]
    return output

def get_corresponding_sentences_in_book(book_output, trans1, trans2):
    """
    Gets corresponding sentences for a given book of the bible for two specified translations.

    :param book_output:
    :param trans1: First translation. e.g. 'WYC'
    :param trans2: Second translation e.g. 'WEB'
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
    corresponding_sentences = []
    for book_title in book_titles:
        book_output = pickle.load(open(get_path("cache/by_book_filtered/{0}.pickle".format(book_title)), "rb"))
        pairs = get_corresponding_sentences_in_book(book_output, trans1, trans2)
        corresponding_sentences.extend(pairs)
    return corresponding_sentences


def get_corresponding_sentences_in_book_multiple(book_output, translations):
    """
    Gets corresponding sentences for a given book of the bible for two specified translations.

    :param book_output:
    :param translations: e.g. ['WYC', 'WEB']
    :return: List of tuples of the verse text, with the first translation first.
    """
    corresponding_sentences = list()
    for chapter_output in book_output.values():
        for verse_output in chapter_output.values():
            if all(t in verse_output for t in translations):
                # TODO: consider whether we want to throw away translations where the verse is empty.
                corresponding_sentences.append([verse_output[t] for t in translations])
    return corresponding_sentences

def get_corresponding_sentences_in_book_multiple_by_title(book_title, translations):
    book_output = pickle.load(open(get_path("cache/by_book_filtered/{0}.pickle".format(book_title)), "rb"))
    return get_corresponding_sentences_in_book_multiple(book_output, translations)

def get_corresponding_sentences_in_bible_multiple(translations):
    """
    Get corresponding sentences in all books of the bible for all translations.
    :param translations:
    :return:
    """
    corresponding_sentences = []
    for book_title in book_titles:
        book_output = pickle.load(open(get_path("cache/by_book_filtered/{0}.pickle".format(book_title)), "rb"))
        pairs = get_corresponding_sentences_in_book_multiple(book_output, translations)
        corresponding_sentences.extend(pairs)
    return corresponding_sentences