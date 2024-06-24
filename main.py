from src import preprocess_data, train, evaluate
from src.train import NaiveBayesClassifier

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
