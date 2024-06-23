from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self, k=1.0):
        self.k = k
        self.word_count_ = defaultdict(lambda: [0, 0])
        self.total_count_ = [0, 0]
        self.word_prob_ = defaultdict(lambda: [0.0, 0.0])

    def train(self, training_data):
        for index, row in training_data.iterrows():
            label = row['label']
            words = row['message']
            label_index = 0 if label == "spam" else 1
            for word in words:
                self.word_count_[word][label_index] += 1
                self.total_count_[label_index] += 1

        for word, count in self.word_count_.items():
            word_prob_if_spam = (self.k + count[1]) / (2 * self.k + self.total_count_[1])
            word_prob_if_non_spam = (self.k + count[0]) / (2 * self.k + self.total_count_[0])

            self.word_prob_[word][1] = word_prob_if_spam
            self.word_prob_[word][0] = word_prob_if_non_spam

    def predict(self, message):
        log_prob_spam = log_prob_non_spam = 0.0

        for word in message:
            if word in self.word_prob_:
                log_prob_spam += math.log(self.word_prob_[word][1])
                log_prob_non_spam += math.log(self.word_prob_[word][0])

        prob_spam = math.exp(log_prob_spam)
        prob_non_spam = math.exp(log_prob_non_spam)

        return 'spam' if prob_spam > prob_non_spam else 'ham'
