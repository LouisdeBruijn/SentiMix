import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

import nltk
import pandas as pd
import time

import operator
from collections import Counter, defaultdict

#read the data from codalab
def read_data(path):
    docs = []
    labels = []
    
    file = open(path, 'r')

    start = True
    sentence = []
    for line in file:
        if line != "\n":
            if start:
                sentence = []
                start = False
                l = line.split()
                #this adds the labels
                labels.append(l[len(l)-1])
            else:
                w = line.strip().split()

                if len(w) < 2:
                    continue

                #only assigning the actual word for now.
                sentence.append(w[0])
        else:
            start = True
            docs.append(sentence)
            continue

    return docs, labels

def identity(x):
    return x

def test(dataset_path, testdata_path = None, split = 0.5, use_cv = False, n_fold = 5):
    X,Y = read_data(dataset_path)
    split_point = int(split * len(X))
    
    if testdata_path == None:
        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]

        Xtest = X[split_point:]
        Ytest = Y[split_point:]
    else:
        Xtrain = X
        Ytrain = Y
        Xtest, Ytest = read_data(testdata_path)
        if use_cv:
            Xtrain = X + Xtest
            Ytrain = Y + Ytest

    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    clfs = []
    clfs.append(Pipeline([('vec', vec), ('cls', MultinomialNB())]))
    clfs.append(Pipeline([('vec', vec), ('cls', KNeighborsClassifier())]))
    clfs.append(Pipeline([('vec', vec), ('cls', DecisionTreeClassifier())]))

    if use_cv:
        for clf in clfs:
            score = cross_val_score(clf, Xtrain, Ytrain, cv=n_fold)
            print(f"{clf}:{score}")
    else:
        return