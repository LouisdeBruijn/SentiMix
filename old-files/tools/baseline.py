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

import json
import nltk
import pandas as pd
import time
import numpy as np

from gensim.models import KeyedVectors

class MeanEmbeddingVectorizer(object):
    def __init__(self, embedding_paths):
        self.embeddings = []
        for paths in embedding_paths:
            print("Loading " + paths)
            self.embeddings.append(KeyedVectors.load_word2vec_format(paths, limit=10000))
        print("Loading finished")

    def fit(self, X, y):
        return self

    def transform(self, X):

        vec = []
        for tokens in X:
            v = word_embedding_vectorizor(tokens, self.embeddings)
            vec.append(v)

        return vec


def word_embedding_vectorizor(doc, embeddings):
    if not isinstance(embeddings, list):
        embeddings = list(embeddings)

    vecs = []

    for token in doc:
        novec = False
        for emb in embeddings:
            try:
                vecs.append(emb[token.lower()])
                novec = False
                break
            except:
                novec = True
                continue
        if novec:
            vecs.append(emb["."])

    vecs = np.array(vecs)
    vecs = np.mean(vecs, axis=0)

    return vecs


def identity(x):
    return x


def test(data, use_cv=False, n_fold=5):
    '''Test all baseline algorithms'''

    clfs = []
    clfs.append((train(MultinomialNB(), data), "Naive bayes"))
    clfs.append((train(DecisionTreeClassifier(), data), "Decision tree"))
    clfs.append((train(LinearSVC(), data), "SVM"))
    clfs.append((train(KNeighborsClassifier(), data), "KNN"))

    for clf in clfs:
        if use_cv:
            score = cross_val_score(
                clf[0], data.x_train, data.y_train, cv=n_fold)
            print(f"{clf[1]}:{score}")
        else:
            clf[0].fit(data.x_train, data.y_train)
            predict = clf[0].predict(data.x_test)

            print(clf[1])
            print(classification_report(data.y_test, predict))
            print("\n")


def test_one(algorithm, data, use_cv=False, n_fold=5):
    '''Test a specific algorithm with custom parameters'''
    Xtrain, Ytrain, Xtest, Ytest = data.output_data()
    clf, vec, clf_object, vec_object, class_labels, coef = train(algorithm, data)
    if use_cv:
        score = cross_val_score(clf, Xtrain, Ytrain, cv=n_fold)
        print(score)
    else:
        predict = clf.predict(Xtest)
        print(classification_report(Ytest, predict))
        print("\n")
        print("Most informative features for the three classes:")
        print_top10(clf_object, vec_object, class_labels)

def print_top10(clf, vectorizer, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def train(algo, data, vec=None):
    '''Fit a model and return it'''
    if vec == None:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        #vec = MeanEmbeddingVectorizer(data.embeddings)

    classifier = Pipeline([('vec', vec), ('cls', algo)])
    classifier.fit(data.x_train, data.y_train)

    cls_object = classifier.named_steps['cls']
    vec_object = classifier.named_steps['vec']
    class_labels = ['negative', 'neutral', 'positive']
    coef = cls_object.coef_

    return classifier, vec, cls_object, vec_object, class_labels, coef


def plot_svm_accuracy(data):
    '''Output a graph of the accuracy of SVM under different c settings'''
    ac = []
    X, Y, XT, YT = data.output_data()
    test_range = 20
    for c in range(1, test_range + 1):
        clf = train(LinearSVC(C=c), data)
        g = clf.predict(XT)
        ac.append(accuracy_score(YT, g))
        print(f"{c/test_range * 100}% complete")

    plt.plot(ac, label="Accuracy")

    plt.title('LinearSVC')
    plt.ylabel('score')
    plt.xlabel('C')
    plt.legend()
    plt.show()


def plot_nb_accuracy(data):
    '''Output a graph of the accuracy of Naive Bayes under different alpha settings'''
    ac = []
    X, Y, XT, YT = data.output_data()
    test_range = 100
    for c in range(1, test_range + 1):
        clf = train(MultinomialNB(alpha=c/100), data)
        g = clf.predict(XT)
        ac.append(accuracy_score(YT, g))
        print(f"{c/test_range * 100}% complete")

    plt.plot(ac, label="Accuracy")

    plt.title('Naive bayes')
    plt.ylabel('score')
    plt.xlabel('alpha/100')
    plt.legend()
    plt.show()
