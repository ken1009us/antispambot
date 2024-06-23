def evaluate(test_data, classifier):
    tp = fp = tn = fn = 0

    for index, row in test_data.iterrows():
        label = row['label']
        message = row['message']
        prediction = classifier.predict(message)
        if prediction == 'spam' and label == 'spam':
            tp += 1
        elif prediction == 'spam' and label == 'ham':
            fp += 1
        elif prediction == 'ham' and label == 'ham':
            tn += 1
        elif prediction == 'ham' and label == 'spam':
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall, (tp, fp, tn, fn)
