from collections import Counter

from data.bible.training_data import read_bible, get_corresponding_sentences_in_bible
from embedder import word2vec
from constants import MAX_GENERATION_SIZE

nirv = list(read_bible(translations=['NIRV']).values())
sents = []
s = []
for sent in nirv:
    for w in sent.split():
        s.append(w)
        if w in '.?!':
            sents.append(s)
            s = []

VOCAB_THRESH = 500 # over all translations
SENT_LEN = 20

common = list(filter(lambda x: all(map(lambda y:y in word2vec and word2vec.vocab[y].count >= VOCAB_THRESH, x)), sents))

# print(sorted(common, key=len)[-1])

short = list(" ".join(sent) for sent in filter(lambda x: len(x) <= SENT_LEN, common))

# print('Vocab size:', len(set([w for s in short for w in s])))
# print('Sentence lengths:', Counter(map(len, short)))
# print(len(sents), len(common), len(short))
pairs = get_corresponding_sentences_in_bible('NIV', 'NIRV')
pairs = list(filter(lambda x:all(map(lambda y: len(y.split()) <= MAX_GENERATION_SIZE, x)), pairs))
common_pairs = list(filter(lambda z: all(map(lambda x: all(map(lambda y:y in word2vec and word2vec.vocab[y].count >= VOCAB_THRESH, x.split())), z)), pairs))
sentences = list(sum(pairs, ()))

"""
niv = list(read_bible(translations=['NIV']).values())
all_trans = read_bible(flatten_translations=True, filter_none=True)
sentence_lengths_niv = Counter(map(lambda x: len(x.split()), niv))
sentence_lengths_all = Counter(map(lambda x: len(x.split()), all_trans))
"""
