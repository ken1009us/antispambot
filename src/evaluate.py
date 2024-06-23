def evaluate(test_X, test_y, classifier):
    tp = fp = tn = fn = 0

    for message, label in zip(test_X, test_y):
        prediction = classifier.predict(message)
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        elif prediction == 0 and label == 1:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall, (tp, fp, tn, fn)
