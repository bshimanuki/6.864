from bs4 import BeautifulSoup
import requests
import pickle

"""
Scrapes bible translations from www.biblestudytools.com. Functions store information in cache folder.
"""

# Sample URLs for testing
verse_url_sample = 'http://www.biblestudytools.com/genesis/1-1-compare.html'
chapter_url_sample = 'http://www.biblestudytools.com/compare-translations/genesis/1/'
book_url_sample = 'http://www.biblestudytools.com/compare-translations/genesis/'
home_url_sample = 'http://www.biblestudytools.com/compare-translations/'

# Spoof header to avoid 403
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'}

def get_child_from_parent(parent_url):
    """
    Gets all links to all the subsections of a section.

    The subsections can either be verses within a chapter, or chapters within a book.

    :param parent_url: URL of either a chapter or a book (of the bible).
    :return: Dictionary mapping title of subsection to URL for subsection.
    """

    def get_hyperlink_content(a):
        url = unicode(a['href'])
        title = unicode(a.string.strip())
        return (title, url)

    response = requests.get(parent_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    panel_body = soup.find("div", { "class" : "panel-body" })
    hyperlinks = panel_body.find_all('a')
    parent_contents = map(get_hyperlink_content, hyperlinks)
    return dict(parent_contents)

def get_book_info():
    """
    Gets links to all books in the bible.

    :return: Dictionary mapping title of book to URL for book.
    """
    home_url = 'http://www.biblestudytools.com/compare-translations/'
    response = requests.get(home_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    hyperlinks = soup.find_all("a", { "class" : "btn btn-lg btn-block bst-button-white" })

    def get_hyperlink_content(a):
        # Slightly different than parallel get_hyperlink_content function above.
        url = unicode(a['href'])
        title = unicode(a.h4.string)
        return (title, url)

    book_contents = map(get_hyperlink_content, hyperlinks)
    return dict(book_contents)

""" {'book_title': {'chapter_title': {'verse_title': {'translation_version': 'text', ...}, ...}, ...}, ...} """

def get_verse_output(verse_url):
    """
    Downloads all available translations of a verse.

    :param verse_url: URL of verse of bible.
    :return: Dictionary mapping translation version to text of verse.
        {'(ASV)': 'In the beginning God created the heavens and the earth',
         '(BBE)': 'At the first God made the heaven and the earth',
         ...}
    """
    response = requests.get(verse_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    divs = soup.findAll("div", { "class" : "scripture" })
    def get_translation_and_text(div):
        link_text = div.strong.string.strip()
        version = link_text.split()[-1] # TODO: maybe use a regex here
        text = div.span.string # NOTE: this could be empty
        plain_text = unicode(text) # changes bs4.element.NavigableString to string
        return (version, plain_text)

    version_to_text = dict(map(get_translation_and_text, divs))
    return version_to_text

def get_chapter_output(chapter_url):
    """

    :param chapter_url: URL of chapter of bible
    :return: Dictionary mapping verse title to return value of get_verse_output
        {'1': {'(ASV)': 'In the beginning God created the heavens and the earth',
               '(BBE)': 'At the first God made the heaven and the earth',
               ...},
         '2': {'(ASV)': 'And the earth was waste and void; ...', ...},
            ...}
    """
    verse_title_to_verse_url = get_child_from_parent(chapter_url)
    chapter_output = dict()
    for verse_title, verse_url in verse_title_to_verse_url.iteritems():
        chapter_output[verse_title] = get_verse_output(verse_url)
    return chapter_output

def get_book_output(book_url):
    """

    :param book_url: URL of book of bible
    :return: Dictionary mapping chapter title to return value of get_chapter_output

    """
    chapter_title_to_chapter_url = get_child_from_parent(book_url)
    book_output = dict()
    for chapter_title, chapter_url in chapter_title_to_chapter_url.iteritems():
        print('  Chapter {0} processing'.format(chapter_title))
        book_output[chapter_title] = get_chapter_output(chapter_url)
    return book_output

def get_all_output():
    """

    :return: Dictionary mapping book title to return value of get_book_output
    """
    book_info = get_book_info()
    out = dict()
    for book_title, book_url in book_info.iteritems():
        print('{0} processing'.format(book_title))
        out[book_title] = get_book_output(book_url)
    return out

def download_all_books():
    """
    Downloads all the books of the bible, caching them as separate pickled files with the book names as titles.
    :return: None
    """
    book_info = get_book_info()
    for book_title, book_url in book_info.iteritems():
        print('{0} processing'.format(book_title))
        book_output = get_book_output(book_url)
        pickle.dump(book_output, open( "cache/by_book/{0}.pickle".format(book_title), "wb" ) )

def download_some_books(required_book_titles):
    """
    Downloads just the subsets of the books you want.
    :param required_book_titles:
    :return:
    """
    book_info = get_book_info()
    for book_title, book_url in book_info.iteritems():
        if book_title in required_book_titles:
            print('{0} processing'.format(book_title))
            book_output = get_book_output(book_url)
            pickle.dump(book_output, open( "cache/by_book/{0}.pickle".format(book_title), "wb" ) )

def get_translation_versions_of_book(book_output):
    """
    Get all the versions of a book in the bible
    :param book_output:
    :return:
    """
    safe_translation_versions = set()
    all_translation_versions = set()
    for chapter_output in book_output.itervalues():
        for verse_output in chapter_output.itervalues():
            safe_translation_versions &= set(verse_output.keys())
            all_translation_versions |= set(verse_output.keys())
    return (safe_translation_versions, all_translation_versions)

def download_translation_versions():
    book_titles = pickle.load(open('cache/book_titles.pickle', 'rb'))
    book_to_stv = dict()
    book_to_atv = dict()
    for book_title in book_titles:
        book_output = pickle.load(open("cache/by_book/{0}.pickle".format(book_title), "rb"))
        (stv, atv) = get_translation_versions_of_book(book_output)
        book_to_stv[book_title] = stv
        book_to_atv[book_title] = atv
    safe_translation_versions = set.intersection(*book_to_atv.values())
    # safe translation versions are guaranteed to be found at least in every book.
    all_translation_versions = set.union(*book_to_atv.values())
    # all translation versions are all the versions we encounter.
    pickle.dump(safe_translation_versions, open("cache/safe_translation_versions.pickle", "wb"))
    pickle.dump(all_translation_versions, open("cache/all_translation_versions.pickle", "wb"))

def download_book_titles():
    book_info = get_book_info()
    book_titles = set(book_info.keys())
    pickle.dump(book_titles, open("cache/book_titles.pickle", "wb"))

# genesis_1 = get_chapter_output('http://www.biblestudytools.com/compare-translations/genesis/1/')
# pickle.dump(genesis_1, open( "cache/genesis_1.pickle", "wb" ) )

# genesis = get_book_output('http://www.biblestudytools.com/compare-translations/genesis/')
# pickle.dump(genesis, open( "cache/Genesis.pickle", "wb" ) )

# download_all_books()

# TODO: Deal with jankiness of cached data: verse and chapter numbers are strings