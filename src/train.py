import math


from .preprocess_data import preprocess_message

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

