from src import preprocess_data, train, evaluate
from src.train import NaiveBayesClassifier
from wordcloud import WordCloud

import matplotlib.pyplot as plt


def plot_message_length_distribution(X, y):
    spam_lengths = [len(msg.split()) for msg, label in zip(X, y) if label == 1]
    non_spam_lengths = [len(msg.split()) for msg, label in zip(X, y) if label == 0]

    plt.figure(figsize=(10, 7))
    plt.hist(spam_lengths, bins=50, alpha=0.5, label='Spam')
    plt.hist(non_spam_lengths, bins=50, alpha=0.5, label='Non-Spam')
    plt.xlabel('Message Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Message Lengths')
    plt.legend()
    plt.show()


def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    filepath = 'data/raw/SMSSpamCollection.txt'
    mails = preprocess_data.read_file(filepath)
    train_X, train_y, test_X, test_y = preprocess_data.split_data(mails)

    training_data = list(zip(train_X, train_y))

    # Define range of k values
    k_values = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    # Find the best k value using cross-validation
    best_k, best_scores = train.find_best_k(training_data, k_values)
    print(f'Best k value: {best_k}')
    print(f'Precision with best k: {best_scores[0]:.2%}, Recall with best k: {best_scores[1]:.2%}')

    # Train final classifier with best k value
    # best_k = 1.0
    classifier = NaiveBayesClassifier(k=best_k)
    classifier.train(training_data)

    # Evaluate on test set
    precision, recall, confusion_matrix = evaluate.evaluate(test_X, test_y, classifier)

    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print('Confusion Matrix:', confusion_matrix)

# =========Visualization===========

    # spam_text = ' '.join([msg for msg, label in zip(train_X, train_y) if label == 1])
    # non_spam_text = ' '.join([msg for msg, label in zip(train_X, train_y) if label == 0])

    # create_wordcloud(spam_text)
    # create_wordcloud(non_spam_text)

    # plot_message_length_distribution(train_X, train_y)

