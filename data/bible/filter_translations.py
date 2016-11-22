import pickle

def remove_unused_translations(unused_translations=['(OJB)', '(TYN)', '(SBLG)']):
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

    :param unused_translations:
    :return:
    """
    book_titles = pickle.load(open('cache/book_titles.pickle', 'rb'))
    for book_title in book_titles:
        book_output = pickle.load(open("cache/by_book/{0}.pickle".format(book_title), "rb"))
        for chapter_output in book_output.values():
            for verse_output in chapter_output.values():
                for t in unused_translations:
                    verse_output.pop(t, None)
        pickle.dump(book_output, open("cache/by_book_filtered/{0}.pickle".format(book_title), "wb"))

remove_unused_translations(['(OJB)', '(TYN)', '(SBLG)', '(HNV)', '(WNT)', '(LXX)'])
