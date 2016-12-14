import numpy as np
from .embedding import Embedding
from .embedder import process_sentence

class OnehotEmbedding(Embedding):
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def words_to_indices(self, token_list):
        return self.word2vec.words_to_indices(token_list)

    def get_embedding_matrix(self):
        vocab = self.word2vec.index_to_word_map()
        num_words = len(vocab)
        identity = np.identity(num_words)
        return identity

    def word_indices(sentences, eos=False):
        sents = [process_sentence(sent, eos) for sent in sentences]
        if eos:
            lens = np.array([len(sent) - 1 for sent in sentences])
        else:
            lens = np.array([len(sent) for sent in sentences])
            max_len = max(lens)
        embeddings = []
        for sent in sentences:
            one_hot_sentence = word2vec.one_hot(sent, max_len)
            embeddings.append(one_hot_sentence)
        return np.array(embeddings), lens

    def embedding_to_sentence(embeddings):
        vocab = word2vec.index_to_word_map()
        word_indices = [word_vec.index(max(word_vec)) for word_vec in embeddings]
        return ' '.join([vocab[index] for index in word_indices])
    
