import re
import pandas as pd
import random


def read_file(filepath):
    print(f"Reading {filepath}...")
    messages = []
    with open(filepath, encoding="utf-8") as file:
        for line in file.readlines():
            is_spam, message = line.split("\t")
            is_spam = 1 if is_spam == "spam" else 0
            message = message.strip()
            messages.append((message, is_spam))
    return messages


def preprocess_message(message):
    message = message.lower()
    words = re.findall("[a-z']+", message)
    return words


def split_data(messages, ratio=0.8):
    print("Splitting data...")
    random.seed(2)
    random.shuffle(messages)
    train_num = round(0.8 * len(messages))
    train_X = [message[0] for message in messages[:train_num]]
    train_y = [message[1] for message in messages[:train_num]]
    test_X = [message[0] for message in messages[train_num:]]
    test_y = [message[1] for message in messages[train_num:]]
    return (train_X, train_y, test_X, test_y)
