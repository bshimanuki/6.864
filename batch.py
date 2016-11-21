from data.bible import training_data
import random
from itertools import chain

class Single:
    def __init__(self):
        a = training_data.get_corresponding_sentences_in_bible('(WYC)', '(WEB)')
        self.a = list(chain.from_iterable(a))
        random.shuffle(self.a)
        self.i = 0

    def next_batch(self, batch_size):
        if self.i + batch_size >= len(self.a):
            self.i = 0
            random.shuffle(self.a)
        ret = self.a[self.i:self.i+batch_size]
        self.i += batch_size
        return ret
