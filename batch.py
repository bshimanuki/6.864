import random
from itertools import chain

class Single:
    def __init__(self, corpus, train_ratio=0.9):
        a = list(corpus)
        random.seed(1337)
        random.shuffle(a)
        t = int(train_ratio * len(a))
        self.a = a[:t]
        self.b = a[t:]
        self.i = 0

    def next_batch(self, batch_size):
        if self.i + batch_size >= len(self.a):
            self.i = 0
            random.shuffle(self.a)
        ret = self.a[self.i:self.i+batch_size]
        self.i += batch_size
        return ret

    def random_validation_batch(self, batch_size):
        random.shuffle(self.b)
        return self.b[:batch_size]

    def num_training(self):
        return len(self.a)

class Pairs:
    def __init__(self, pair_corpus, train_ratio=0.9):
        a = list(pair_corpus)
        random.seed(1337)
        random.shuffle(a)
        t = int(train_ratio * len(a))
        self.a = a[:t]
        self.b = a[t:]
        self.i = 0

    def next_batch(self, batch_size):
        if self.i + batch_size >= len(self.a):
            self.i = 0
            random.shuffle(self.a)
        out = [list(t) for t in zip(*self.a[self.i:self.i+batch_size])]
        self.i += batch_size
        return out[0], out[1]

    def random_validation_batch(self, batch_size):
        random.shuffle(self.b)
        out = [list(t) for t in zip(*self.b[:batch_size])]
        return out[0], out[1]

    def num_training(self):
        return len(self.a)

class Quads:
    """Use for a pair corpus only. Will need to rewrite for multiple translations."""
    def __init__(self, pair_corpus, train_ratio=0.9):
        a = list(pair_corpus)
        random.seed(1337)
        random.shuffle(a)
        t = int(train_ratio * len(a))
        self.a = a[:t]
        self.b = a[t:]
        self.i = 0

    def next_batch(self, batch_size):
        if self.i + 2*batch_size >= len(self.a):
            self.i = 0
            random.shuffle(self.a)
        out = [list(t) for t in zip(*self.a[self.i:self.i+batch_size])]
        out2 = [list(t) for t in zip(*self.a[self.i+batch_size:self.i+2*batch_size])]
        self.i += 2*batch_size
        print(out)
        return out[0], out[1], out2[0], out2[1]

    def random_validation_batch(self, batch_size):
        random.shuffle(self.b)
        out = [list(t) for t in zip(*self.a[:batch_size])]
        out2 = [list(t) for t in zip(*self.a[batch_size:2*batch_size])]
        return out[0], out[1], out2[0], out2[1]

    def num_training(self):
        return len(self.a) / 2
