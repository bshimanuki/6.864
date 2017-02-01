import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, RANDOM_SEED
from data.bible.training_data import get_corresponding_sentences_in_bible as get_pairs
from data.bible.training_data import get_corresponding_sentences_in_book_multiple_by_title
from embedding.embedder import word2vec
from pair_nn import get_hidden

# WYC-WEB 0.97 accuracy
#trans_pairs = get_pairs('WYC', 'WEB')

# ASV-CEB 0.95 accuracy
# trans_pairs = get_pairs('ASV', 'CEB')

# trans_pairs = get_corresponding_sentences_in_bible_multiple(['ASV', 'CEB', 'WYC', 'WEB'])

use_svm = True
def _get_classifier(c):
    if use_svm:
        return SVC(C=c, kernel='rbf', gamma=0.005)
    else:
        return LogisticRegression(C=c)

def get_corpus(trans1, trans2):
    vocab = set()
    trans_pairs = get_pairs(trans1, trans2)
    trans_pairs = [[translation.split(' ') for translation in pair] for pair in trans_pairs]

    for trans_pair in trans_pairs:
        for translation in trans_pair:
            for word in translation:
                vocab.add(word)

    return trans_pairs, vocab

def perform_split(feature_vectors, labels):
    train_val_feature_vectors, test_feature_vectors, train_val_labels, test_labels = train_test_split(
        feature_vectors, labels, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    scaler = StandardScaler().fit(train_val_feature_vectors)
    train_val_feature_vectors = scaler.transform(train_val_feature_vectors)
    test_feature_vectors = scaler.transform(test_feature_vectors)

    test_feature_vectors = sparse.csr_matrix(test_feature_vectors)

    train_feature_vectors, validation_feature_vectors, train_labels, validation_labels = train_test_split(
        train_val_feature_vectors, train_val_labels, test_size=(VALIDATION_RATIO)/(VALIDATION_RATIO+TRAIN_RATIO), random_state=RANDOM_SEED
    )

    train_val_feature_vectors = sparse.csr_matrix(train_val_feature_vectors)
    train_feature_vectors = sparse.csr_matrix(train_feature_vectors)
    validation_feature_vectors = sparse.csr_matrix(validation_feature_vectors)

    print("Split dataset into training, test and validation sets.")
    print("Training size: {}".format(np.shape(train_feature_vectors)))
    print("Validation size: {}".format(np.shape(validation_feature_vectors)))
    print("Test size: {}".format(np.shape(test_feature_vectors)))
    return train_feature_vectors, validation_feature_vectors, train_val_feature_vectors, test_feature_vectors, train_labels, validation_labels, train_val_labels, test_labels

def get_unigram_features(trans1, trans2, use_unk=True):
    trans_pairs, vocab = get_corpus(trans1, trans2)
    word_to_index = {}
    num_words = 1
    for word in vocab:
        if use_unk and word not in word2vec.vocab:
            word_to_index[word] = 0
        else:
            word_to_index[word] = num_words
            num_words += 1

    print("Extracted vocabulary and mapping.")

    # potentially speed this up by using a sparse representation.
    feature_vectors = []
    labels = []
    for trans_pair in trans_pairs:
        for i, translation in enumerate(trans_pair):
            unigram_features = [0.0]*num_words
            for word in translation:
                if word in vocab:
                    unigram_features[word_to_index[word]] = 1.0
            feature_vectors.append(unigram_features)
            labels.append(i)

    print("Converted sentences to feature vectors and labels.")
    print("\tLabels: {}".format(np.array(labels)))

    ret = perform_split(feature_vectors, labels)
    del feature_vectors
    del labels
    return ret

def get_nn_features():
    features1, features2 = get_hidden(["style", "logvar_style"])
    num_sents, num_feats = features1.shape
    labels = np.concatenate((np.array([0]*num_sents), np.array([1]*num_sents)), axis=0)
    features = np.concatenate((features1, features2), axis=0)
    features_labels = np.concatenate((features, np.expand_dims(labels, axis=1)), axis=1)
    assert features_labels.shape == (num_sents*2, num_feats+1)
    np.random.shuffle(features_labels)
    feature_vectors = features_labels[:,:-1]
    labels = features_labels[:,-1]

    print("Converted sentences to feature vectors and labels.")
    print("\tLabels: {}".format(np.array(labels)))

    ret = perform_split(feature_vectors, labels)
    del feature_vectors
    del labels
    return ret

def check_matches(labels, predicted_labels):
    if np.size(labels) != np.size(predicted_labels):
        raise ValueError
    num_matches = np.sum(np.array(labels) == np.array(predicted_labels))
    proportion_matched = num_matches / np.size(labels)
    return (num_matches, proportion_matched)

def evaluate_pairs(trans1, trans2, type='unigram'):
    if type == 'unigram':
        process_inputs = get_unigram_features(trans1, trans2)
    else:
        process_inputs = get_nn_features()
    process(*process_inputs)

def process(train_feature_vectors, validation_feature_vectors, train_val_feature_vectors, test_feature_vectors, train_labels, validation_labels, train_val_labels, test_labels):
    print("Using validation set to optimize over value of regularization parameter in regression, C.")
    best_acc = 0
    C_RANGE = [0.1, 1, 10]
    for C_cur in C_RANGE:
        """
        It turns out that the default value is pretty good, with performance smoothly increasing then smoothly
        decreasing after 1. You can choose to verify this by passing in
        C_RANGE = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
        """
        print("\tCurrent C: {}".format(C_cur))
        classifier = _get_classifier(C_cur)
        classifier.fit(train_feature_vectors, train_labels)
        predicted_train_labels = classifier.predict(train_feature_vectors)
        predicted_validation_labels = classifier.predict(validation_feature_vectors)
        (_, train_acc) = check_matches(train_labels, predicted_train_labels)
        (_, val_acc) = check_matches(validation_labels, predicted_validation_labels)
        print("\t\tTraining accuracy: {}".format(train_acc))
        print("\t\tValidation accuracy: {}".format(val_acc))
        if val_acc > best_acc:
            best_C = C_cur
            best_acc = val_acc

    print("Done training and validating. Best C found: {}, Best accuracy on validation: {}".format(best_C, best_acc))

    classifier = _get_classifier(best_C)
    classifier.fit(train_val_feature_vectors, train_val_labels)
    predicted_train_val_labels = classifier.predict(train_val_feature_vectors)
    predicted_test_labels = classifier.predict(test_feature_vectors)
    (num_matches_train, accuracy_train) = check_matches(train_val_labels, predicted_train_val_labels)
    (num_matches_test, accuracy_test) = check_matches(test_labels, predicted_test_labels)
    print("Accuracy on train {}".format(accuracy_train))
    print("Accuracy on test {}".format(accuracy_test))

if __name__ == '__main__':
    evaluate_pairs('NIV', 'NIRV', type='nn')
