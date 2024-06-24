from .preprocess_data import preprocess_message
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score

import numpy as np
import math

def evaluate_classifier(k, training_data, validation_data):
    classifier = NaiveBayesClassifier(k=k)
    classifier.train(training_data)
    validation_X, validation_y = zip(*validation_data)
    predictions = [classifier.predict(x) for x in validation_X]

    precision = precision_score(validation_y, predictions)
    recall = recall_score(validation_y, predictions)

    return precision, recall

def cross_validate_k(training_data, k_values, n_splits=5):
    kf = KFold(n_splits=n_splits)
    results = {k: [] for k in k_values}

    for train_index, val_index in kf.split(training_data):
        train_data = [training_data[i] for i in train_index]
        val_data = [training_data[i] for i in val_index]

        for k in k_values:
            precision, recall = evaluate_classifier(k, train_data, val_data)
            results[k].append((precision, recall))

    avg_results = {k: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for k, v in results.items()}
    return avg_results

def find_best_k(training_data, k_values):
    avg_results = cross_validate_k(training_data, k_values)
    best_k = max(avg_results, key=lambda k: avg_results[k][0])
    return best_k, avg_results[best_k]


class NaiveBayesClassifier:
    def __init__(self, k=1.0):
        self.k = k
        self.word_count_ = {}
        self.total_count_ = [0, 0]
        self.word_prob_ = {}

    def train(self, training_data):
        for message, label in training_data:
            words = preprocess_message(message)
            label_index = 1 if label == 1 else 0
            for word in words:
                if word not in self.word_count_:
                    self.word_count_[word] = [0, 0]
                self.word_count_[word][label_index] += 1
                self.total_count_[label_index] += 1

        for word, count in self.word_count_.items():
            word_prob_if_spam = (self.k + count[1]) / (self.k * 2 + self.total_count_[1])
            word_prob_if_non_spam = (self.k + count[0]) / (self.k * 2 + self.total_count_[0])

            if word not in self.word_prob_:
                self.word_prob_[word] = [0.0, 0.0]

            self.word_prob_[word][1] = word_prob_if_spam
            self.word_prob_[word][0] = word_prob_if_non_spam

    def predict(self, message):
        words = preprocess_message(message)
        log_prob_spam = log_prob_non_spam = 0.0

        for word in words:
            if word in self.word_prob_:
                log_prob_spam += math.log(self.word_prob_[word][1])
                log_prob_non_spam += math.log(self.word_prob_[word][0])

        prob_spam = math.exp(log_prob_spam)
        prob_non_spam = math.exp(log_prob_non_spam)

        total_prob = prob_spam + prob_non_spam

        if total_prob == 0:
            return 0

        spam_probability = prob_spam / total_prob
        return 1 if spam_probability > 0.5 else 0

