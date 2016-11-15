from bs4 import BeautifulSoup
import requests
import pickle

verse_url = 'http://www.biblestudytools.com/genesis/1-1-compare.html'
chapter_url = 'http://www.biblestudytools.com/compare-translations/genesis/1/'
book_url = 'http://www.biblestudytools.com/compare-translations/genesis/'
home_url = 'http://www.biblestudytools.com/compare-translations/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'}
# spoof header to avoid 403

def get_verse_output(verse_url):
    response = requests.get(verse_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    divs = soup.findAll("div", { "class" : "scripture" })
    def get_translation_and_text(div):
        link_text = div.strong.string.strip()
        version = link_text.split()[-1] # TODO: maybe use a regex here
        text = div.span.string # NOTE: this could be empty
        plain_text = unicode(text) # change bs4.element.NavigableString to string
        return (version, plain_text)

    version_to_text = dict(map(get_translation_and_text, divs))
    return version_to_text

    # verse_link = map(lambda div: div.strong.string.strip(), mydivs)
    # verse_translation = map(lambda link: link.split()[-1], verse_link) # TODO: maybe use a regex here
    # verse_text = map(lambda div: div.span.string.strip(), mydivs)
    # relevant bits: soup.h1

def get_child_from_parent(parent_url):
    """

    :param parent_url: Can be either chapter_url or book_url
    :return:
    """
    # TODO: probably can search by a

    def get_hyperlink_content(a):
        url = a['href']
        title = a.string.strip()
        return (title, url)

    response = requests.get(parent_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup.prettify())
    panel_body = soup.find("div", { "class" : "panel-body" })
    hyperlinks = panel_body.find_all('a')
    # child_urls = map(lambda a: a['href'], hyperlinks)
    parent_contents = map(get_hyperlink_content, hyperlinks)
    return dict(parent_contents)

def get_book_info():
    home_url = 'http://www.biblestudytools.com/compare-translations/'
    response = requests.get(home_url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    # print(soup.prettify())
    hyperlinks = soup.find_all("a", { "class" : "btn btn-lg btn-block bst-button-white" })

    def get_hyperlink_content(a):
        # works only for the home url
        url = a['href']
        title = a.h4.string
        return (title, url)

    # book_urls = map(lambda a: a['href'], hyperlinks)
    book_contents = map(get_hyperlink_content, hyperlinks)
    return dict(book_contents)

""" {'book_title': {'chapter_title': {'verse_title': {'translation_version': 'text', ...}, ...}, ...}, ...} """

def get_chapter_output(chapter_url):
    verse_title_to_verse_url = get_child_from_parent(chapter_url)
    chapter_output = dict()
    for verse_title, verse_url in verse_title_to_verse_url.iteritems():
        chapter_output[verse_title] = get_verse_output(verse_url)
    return chapter_output

def get_book_output(book_url):
    chapter_title_to_chapter_url = get_child_from_parent(book_url)
    book_output = dict()
    for chapter_title, chapter_url in chapter_title_to_chapter_url.iteritems():
        print('  Chapter {0} processing'.format(chapter_title))
        book_output[chapter_title] = get_chapter_output(chapter_url)
    return book_output

def get_all_info():
    book_info = get_book_info()
    out = dict()
    for book_title, book_url in book_info.iteritems():
        print('{0} processing'.format(book_title))
        out[book_title] = get_book_output(book_url)
    return out

def do_all_info():
    # side effecty, does the caching
    book_info = get_book_info()
    out = dict()
    for book_title, book_url in book_info.iteritems():
        print('{0} processing'.format(book_title))
        book_output = get_book_output(book_url)
        pickle.dump(book_output, open( "cache/{0}.pickle".format(book_title), "wb" ) )

# genesis_1 = get_chapter_output('http://www.biblestudytools.com/compare-translations/genesis/1/')
# pickle.dump(genesis_1, open( "cache/genesis_1.pickle", "wb" ) )

# genesis = get_book_output('http://www.biblestudytools.com/compare-translations/genesis/')
# pickle.dump(genesis, open( "cache/genesis.pickle", "wb" ) )

do_all_info()