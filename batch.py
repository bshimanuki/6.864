from nirv import short
import random
from itertools import chain

class Single:
    def __init__(self):
        self.a = list(" ".join(sent) for sent in short)
        random.shuffle(self.a)
        self.i = 0

    def next_batch(self, batch_size):
        if self.i + batch_size >= len(self.a):
            self.i = 0
            random.shuffle(self.a)
        ret = self.a[self.i:self.i+batch_size]
        self.i += batch_size
        return ret
