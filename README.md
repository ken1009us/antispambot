# ANTI-SPAMBOT

This project implements a Naive Bayes classifier to detect the spam messages. The classifier is based on Bayes' Theorem, which provides a probabilistic approach to classify messages as spam or non-spam.

## Why Use Bayes’ Theorem?

Bayes' Theorem is a powerful tool in probability theory and statistics. It describes the probability of an event, based on prior knowledge of conditions that might be related to the event. In the context of spam detection, Bayes' Theorem allows us to update the probability estimate of a message being spam based on the presence of certain words in the message.

## How the Project Was Completed

### Step 1: Data Preprocessing

- Reading Data: The dataset was read from a text file where each line contains a message and its label (spam or non-spam).

- Tokenization: Each message was converted to lowercase and split into words (tokens) using regular expressions.

### Step 2: Training the Naive Bayes Classifier

- Word Frequency Calculation: For each word, its frequency in spam and non-spam messages was calculated.

- Probability Calculation: The conditional probabilities of each word being in a spam or non-spam message were computed using Laplace smoothing to handle zero frequencies.

### Step 3: Prediction

- Log Probability Calculation: For a given message, the log probabilities of the message being spam or non-spam were calculated by summing the log probabilities of individual words.

- Classification: The message was classified as spam if the probability of being spam was higher than that of being non-spam.

### Step 4: Evaluation

- Precision and Recall: The model's performance was evaluated using precision and recall metrics, as well as a confusion matrix to understand the classification results.

## Results

The final model achieved the following results:

```bash
|               | Predicted Ham | Predicted Spam |
|---------------|---------------|----------------|
| Actual Ham    |      913      |       46       |
| Actual Spam   |       4       |      152       |
|---------------|---------------|----------------|

Precision: 76.77%
Recall: 97.44%
Confusion Matrix: (152, 46, 913, 4)
```

## Improvement

### Finding the optimal Laplace smoothing factor (k) using cross-validation

In order to determine the optimal k value (Laplace smoothing factor) for my classifier, I try to use a method like cross-validation to evaluate the performance of different k values. This will help me choose the k value that yields the best performance on the dataset.

```bash
Best k value: 0.001
Precision with best k: 91.81%, Recall with best k: 93.54%
Precision: 95.51%
Recall: 95.51%
Confusion Matrix: (149, 7, 952, 7)
```

## Visualizations

1. Distribution of Message Lengths:

- Compare the length distribution of spam vs non-spam messages

![image](https://github.com/ken1009us/antispambot/blob/main/img/messagelength.png "message length")

2. Word Cloud

- Visualize the most common words in spam and non-spam messages

Spam word cloud:

![image](https://github.com/ken1009us/antispambot/blob/main/img/spamwordcloud.png "spam word cloud")

Non-spam word cloud:


![image](https://github.com/ken1009us/antispambot/blob/main/img/nonspamwordcloud.png "non-spam word cloud")


## How to Use

- Preprocess Data

```bash
python scripts/preprocess_data.py
```

- Train the Model

```bash
python scripts/train_model.py
```

- Predict Spam

```bash
python scripts/predict_spam.py
```

- Evaluate the Model

```bash
python scripts/evaluate_model.py
```

## Application in Other Industries

Bayes' Theorem can be applied in various industries beyond spam detection. For example, in baseball, it can be used for:

- Player Performance Prediction
    - Scenario: Predicting a player's performance based on their past performance data and current conditions (e.g., weather, opponent's strengths).

- Method: Calculate the probability of a player achieving a certain performance level (e.g., batting average) given the current conditions.

- Injury Risk Assessment
    - Scenario: Estimating the probability of player injuries based on historical injury data and current physical condition metrics.

    - Method: Use historical data to update the probability of injuries given specific indicators (e.g., player fatigue, recent injury history).

- Game Strategy Optimization
    - Scenario: Deciding the optimal strategy for a game based on the probability of different outcomes.

    - Method: Update the probabilities of winning using current game conditions and opponent's strategies.

By leveraging Bayes' Theorem, baseball teams can make more informed decisions and optimize their strategies based on probabilistic insights.

## Reference

1. SMS Spam Collection, UC Irvine, Available from: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

2. Bayes' theorem, wiki, Available from: https://en.wikipedia.org/wiki/Bayes%27_theorem

3. Improve Naive Bayes Text classifier using Laplace Smoothing, CHIRAG GOYAL, Available from: https://www.analyticsvidhya.com/blog/2021/04/improve-naive-bayes-text-classifier-using-laplace-smoothing:

4. Bayes’ Theorem in email spam filtering, Cornell University, Available from: https://blogs.cornell.edu/info2040/2018/10/27/bayes-theorem-in-email-spam-filtering/

5. “Demystifying Naïve Bayes: Log Probability and Laplace Smoothing”, ajaymehta, Available from: https://medium.com/@dancerworld60/demystifying-na%C3%AFve-bayes-log-probability-and-laplace-smoothing-d6da61b0e70b

