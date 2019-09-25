import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

import nltk
import pandas as pd
import time
from baseline_tools.data import Data


def identity(x):
    return x


def test(dataset_path, testdata_path=None, split=0.5, use_cv=False, n_fold=5):
    '''Test all baseline algorithms'''
    data = Data(
        dataset_path, testdata_path, split, use_cv)

    clfs = []
    clfs.append((train(MultinomialNB(), Xtrain, Ytrain), "Naive bayes"))
    clfs.append((train(LinearSVC(), Xtrain, Ytrain), "Linear SVC"))
    clfs.append(
        (train(DecisionTreeClassifier(), Xtrain, Ytrain), "Decision Tree"))
    clfs.append(
        (train(KNeighborsClassifier(n_neighbors=10), Xtrain, Ytrain), "KNN"))

    for clf in clfs:
        if use_cv:
            score = cross_val_score(clf[0], Xtrain, Ytrain, cv=n_fold)
            print(f"{clf[1]}:{score}")
        else:
            clf[0].fit(Xtrain, Ytrain)
            predict = clf[0].predict(Xtest)

            print(clf[1])
            print(classification_report(Ytest, predict))
            print("\n")


def test_one(algorithm, dataset_path, testdata_path=None, split=0.5, use_cv=False, n_fold=5):
    '''Test a specific algorithm with custom parameters'''
    Xtrain, Ytrain, Xtest, Ytest = get_data(
        dataset_path, testdata_path, split, use_cv)
    clf = train(algorithm, Xtrain, Ytrain)
    if use_cv:
        score = cross_val_score(clf, Xtrain, Ytrain, cv=n_fold)
        print(score)
    else:
        predict = clf.predict(Xtest)
        print(classification_report(Ytest, predict))
        print("\n")


def train(algo, Xtrain, Ytrain):
    '''Fit a model and return it'''
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    classifier = Pipeline([('vec', vec), ('cls', algo)])
    classifier.fit(Xtrain, Ytrain)
    return classifier


def plot_svm_accuracy(datapath):
    '''Output a graph of the accuracy of SVM under different c settings'''
    ac = []
    X, Y, XT, YT = get_data(datapath)
    test_range = 20
    for c in range(1, test_range + 1):
        clf = train(LinearSVC(C=c), X, Y)
        g = clf.predict(XT)
        ac.append(accuracy_score(YT, g))
        print(f"{c/test_range * 100}% complete")

    plt.plot(ac, label="Accuracy")

    plt.title('LinearSVC')
    plt.ylabel('score')
    plt.xlabel('C')
    plt.legend()
    plt.show()


def plot_nb_accuracy(datapath):
    '''Output a graph of the accuracy of Naive Bayes under different alpha settings'''
    ac = []
    X, Y, XT, YT = get_data(datapath)
    test_range = 100
    for c in range(1, test_range + 1):
        clf = train(MultinomialNB(alpha=c/100), X, Y)
        g = clf.predict(XT)
        ac.append(accuracy_score(YT, g))
        print(f"{c/test_range * 100}% complete")

    plt.plot(ac, label="Accuracy")

    plt.title('Naive bayes')
    plt.ylabel('score')
    plt.xlabel('alpha/100')
    plt.legend()
    plt.show()
