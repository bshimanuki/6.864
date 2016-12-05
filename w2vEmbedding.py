from embedding import Embedding
import numpy as np
from embedder import process_sentence

class W2VEmbedding(Embedding):
    def __init__(self, word2vec):
        self.word2vec = word2vec

    def words_to_indices(self, token_list):
        return self.word2vec.words_to_indices(token_list)

    def get_embedding_matrix(self):
        return self.word2vec.embedding_matrix()
    
    def word_indices(self, sentences, eos=False):
        if eos:
            sents = [process_sentence(sent, eos=True) for sent in sentences]
            lens = np.array([len(sent) - 1 for sent in sents])
            max_len = max(lens) + 1
        else:
            sents = [process_sentence(sent) for sent in sentences]
            lens = np.array([len(sent) for sent in sents])
            max_len = max(lens)
        v = np.stack([self.word2vec.words_to_indices(sent, sent_length=max_len) for sent in sents], axis=0)
        return v, lens

    def embedding_to_sentence(self, embeddings):
        vocab = self.word2vec.index_to_word_map()
        word_probs = np.matmul(embeddings, np.transpose(self.get_embedding_matrix()))
        num_words_sentence, num_words_vocab = word_probs.shape
        word_sequence = [vocab[np.argmax(word_probs[i])] for i in range(num_words_sentence)]
        return ' '.join(word_sequence)
