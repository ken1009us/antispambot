import re

def predict_message(classifier, message):
    words = re.findall(r"[a-z']+", message.lower())
    return classifier.predict(words)
