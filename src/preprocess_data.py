import re
import pandas as pd


from sklearn.model_selection import train_test_split


def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            label, message = line.strip().split('\t')
            data.append((label, message))
    return pd.DataFrame(data, columns=['label', 'message'])


def preprocess_data(df):
    df['message'] = df['message'].apply(lambda x: x.lower())
    df['message'] = df['message'].apply(lambda x: re.findall("[a-z']+", x))
    return df


def split_data(df, test_size=0.2):
    return train_test_split(df, test_size=test_size, random_state=42, stratify=df['label'])
