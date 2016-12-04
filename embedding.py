import abc

class Embedding:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def words_to_indices(self, token_list):
        pass

    @abc.abstractmethod
    def get_embedding_matrix(self):
        pass

    @abc.abstractmethod
    def word_indices(self, sentences, eos=False):
        pass

    @abc.abstractmethod
    def embedding_to_sentence(self, embeddings):
        pass

    def get_eos_embedding(self):
        return self.get_embedding_matrix()[self.words_to_indices(['<EOS>'])[0]]

    def get_num_features(self):
        _, num_features = self.get_embedding_matrix().shape
        return num_features

    def get_vocabulary_size(self):
        num_words, _ = self.get_embedding_matrix().shape
        return num_words