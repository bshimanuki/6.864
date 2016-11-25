import os
import pickle

from .book_titles import book_titles
from .translation_versions import translation_versions

def get_path(rel_path):
    return os.path.join(os.path.dirname(__file__), rel_path)

def read_bible(books=[], chapters=[], verses=[], translations=[],
        flatten_books=True, flatten_chapters=True, flatten_verses=True,
        flatten_translations=False):
    """Reads the bible verses and translations from the pickles, returning a list (if flatten_* are all True) or a recursive dict with the flatten_*=False as keys.

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
                    sent = sent.split()
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

