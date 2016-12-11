from collections import Counter

from data.bible.training_data import read_bible, get_corresponding_sentences_in_bible
from embedder import word2vec

nirv = read_bible()['NIRV']
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
PAIR_SENT_LEN = 50

common = list(filter(lambda x: all(map(lambda y:y in word2vec and word2vec.vocab[y].count >= VOCAB_THRESH, x)), sents))

# print(sorted(common, key=len)[-1])

short = list(" ".join(sent) for sent in filter(lambda x: len(x) <= SENT_LEN, common))

# print('Vocab size:', len(set([w for s in short for w in s])))
# print('Sentence lengths:', Counter(map(len, short)))
# print(len(sents), len(common), len(short))
pairs = get_corresponding_sentences_in_bible('NIV', 'NIRV')
sentences = list(sum(pairs, ()))

short_pairs = list(filter(lambda x:all(map(lambda y: len(y.split()) <= PAIR_SENT_LEN, x)), pairs))
