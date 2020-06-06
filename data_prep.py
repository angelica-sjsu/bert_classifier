#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# there is a balanced number of  positive and negative reviews! 
def splitter(X, y):
    """
    :param X: contains all reviews from the dataset
    :param y: contains all senetiments -- acts as target label
    :return: 4 series that will contain train: reviews and labes, test: reviews and labels
    """
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, y_train, X_test, y_test


# BERT will be needing numerical values: transforming the labels into 0(negative) and 1(positive)
def encoder(labels):
    """
    :params labels: binary labels for each review
    :return ecoded_labels: transformed labels into 0 and 1
    """
    # LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    
    print(f'[STATUS] -- encoding labels....')
    print(f'Classes: {label_encoder.classes_}')
    
    encoded_labels = label_encoder.transform(labels)
    print(f'[STATUS] -- labels encoded: {encoded_labels}')
    
    return encoded_labels


# transforming data into the format accepted by BERT
def bert_formatter(X, y):
    """
    :params X: reviews 
    :params y: numerical sentiment value for each review
    :return BERT friendly data frame
    """
    bert_data = pd.DataFrame({
        'label': y,
        'texts': X.replace(r'\n', ' ', regex=True)
    })
    
    return bert_data




