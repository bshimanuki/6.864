from data.bible.training_data import get_corresponding_sentences_in_bible as get_pairs
import numpy as np
from sklearn.linear_model import LogisticRegression
from constants import TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO

vocab = set()
#WYC-WEB 0.997 accuracy
sent_pairs = get_pairs('WYC', 'WEB')

# ASV-CEB 0.998 accuracy
#sent_pairs = get_pairs('ASV', 'CEB')
sent_pairs = [[translation.split(' ') for translation in sentence] for sentence in sent_pairs]

for sentence in sent_pairs:
    for translation in sentence:
        for word in translation:
            vocab.add(word)

num_words = len(vocab)

word_to_index = {}
for i, word in enumerate(vocab):
    word_to_index[word] = i

features = []
labels = []
for sentence in sent_pairs:
    for i, translation in enumerate(sentence):
        unigram_features = [0.0]*num_words
        for word in translation:
            if word in vocab:
                unigram_features[word_to_index[word]] = 1.0
        features.append(unigram_features)
        labels.append(i)

train_index = int(num_words * TRAIN_RATIO)
validation_index = int(num_words * VALIDATION_RATIO) + train_index

train_features = features[0:train_index]
validation_features = features[train_index:validation_index]
test_features = features[validation_index:-1]
train_labels = labels[0:train_index]
validation_labels = labels[train_index:validation_index]
test_labels = labels[validation_index:-1]

best_loss = len(validation_labels)
for C in [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]:
    classifier = LogisticRegression(C=1e5)
    classifier.fit(train_features, train_labels)
    predicted_labels = classifier.predict(validation_features)
    loss = np.array(validation_labels) - np.array(predicted_labels)
    loss = np.linalg.norm(loss)
    if loss < best_loss:
        best_C = C
        best_loss = loss

print("Done training and validating. Best C found: " + str(best_C))

classifier = LogisticRegression(C=best_C)
classifier.fit(np.concatenate((train_features, validation_features)), np.concatenate((train_labels, validation_labels)))
predicted_labels = classifier.predict(test_features)
loss = np.array(test_labels) - np.array(predicted_labels)
loss = np.linalg.norm(loss)/float(len(test_labels))
accuracy = 1.0 - loss
print(accuracy)
