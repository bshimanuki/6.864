from data.bible.training_data import get_corresponding_sentences_in_bible as get_pairs
from data.bible.training_data import get_corresponding_sentences_in_bible_multiple
from scipy import sparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from constants import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO, RANDOM_SEED

vocab = set()

# WYC-WEB 0.97 accuracy
trans_pairs = get_pairs('WYC', 'WEB')

# ASV-CEB 0.95 accuracy
# trans_pairs = get_pairs('ASV', 'CEB')

# trans_pairs = get_corresponding_sentences_in_bible_multiple(['ASV', 'CEB', 'WYC', 'WEB'])
trans_pairs = [[translation.split(' ') for translation in pair] for pair in trans_pairs]

for trans_pair in trans_pairs:
    for translation in trans_pair:
        for word in translation:
            vocab.add(word)

num_words = len(vocab)

word_to_index = {}
for i, word in enumerate(vocab):
    word_to_index[word] = i

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

train_val_feature_vectors, test_feature_vectors, train_val_labels, test_labels = train_test_split(
    feature_vectors, labels, test_size=TEST_RATIO, random_state=RANDOM_SEED
)

del feature_vectors
del labels
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

def check_matches(labels, predicted_labels):
    if np.size(labels) != np.size(predicted_labels):
        raise ValueError
    num_matches = np.sum(np.array(labels) == np.array(predicted_labels))
    proportion_matched = num_matches / np.size(labels)
    return (num_matches, proportion_matched)

print("Using validation set to optimize over value of regularization parameter in logistic regression, C.")
best_proportion_matched = 0
C_RANGE = [0.1, 1, 10]
for C_cur in C_RANGE:
    """
    It turns out that the default value is pretty good, with performance smoothly increasing then smoothly
    decreasing after 1. You can choose to verify this by passing in
    C_RANGE = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
    """
    print("\tCurrent C: {}".format(C_cur))
    classifier = LogisticRegression(C=C_cur)
    classifier.fit(train_feature_vectors, train_labels)
    predicted_validation_labels = classifier.predict(validation_feature_vectors)
    (num_matches, proportion_matched) = check_matches(validation_labels, predicted_validation_labels)
    print("\t\tNumber of matches: {}".format(num_matches))
    print("\t\tProportion of matches: {}".format(proportion_matched))
    if proportion_matched > best_proportion_matched:
        best_C = C_cur
        best_proportion_matched = proportion_matched

print("Done training and validating. Best C found: {}, Best accuracy on validation: {}".format(best_C, best_proportion_matched))

classifier = LogisticRegression(C=best_C)
classifier.fit(train_val_feature_vectors, train_val_labels)
predicted_test_labels = classifier.predict(test_feature_vectors)
(num_matches, accuracy) = check_matches(test_labels, predicted_test_labels)
print("Accuracy on test {}".format(accuracy))
