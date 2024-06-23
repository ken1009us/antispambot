from src import preprocess_data, train, evaluate



if __name__ == "__main__":
    filepath = 'data/raw/SMSSpamCollection.txt'
    mails = preprocess_data.read_file(filepath)
    train_X, train_y, test_X, test_y = preprocess_data.split_data(mails)

    training_data = list(zip(train_X, train_y))

    classifier = train.NaiveBayesClassifier()
    classifier.train(training_data)

    precision, recall, confusion_matrix = evaluate.evaluate(test_X, test_y, classifier)

    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print('Confusion Matrix:', confusion_matrix)