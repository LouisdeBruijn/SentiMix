import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data.data_manager import Data, Preprocessor, Explorer
import random


class MeanEmbeddingVectorizer(object):
    def __init__(self, embs: list):
        self.embs = embs
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):

        vec = []
        for tokens in X:
            vectors = []
            for token in tokens:
                try:
                    emb_vec = self.embs[0]["unk"]
                except KeyError:
                    emb_vec = self.embs[0]["UNK"]

                for emb in self.embs:
                    try:
                        emb_vec = emb[token.lower()]
                    except KeyError:
                        continue
                vectors.append(emb_vec)

            v = np.array(vectors)
            v = np.mean(v, axis=0)
            vec.append(v)

        return vec


def run_random_baseline(traindata: Data, testdata: Data):
    posl = traindata.labels.count("positive")
    negl = traindata.labels.count("negative")
    neul = traindata.labels.count("neutral")

    pred = []

    for doc in testdata.labels:
        num = random.uniform(0, len(traindata.labels))
        if num >= 0 and num <= posl:
            pred.append("positive")
        elif num > posl and num <= negl + posl:
            pred.append("negative")
        else:
            pred.append("neutral")

    ytest = testdata.labels
    print(classification_report(ytest, pred))
    cm = confusion_matrix(ytest, pred)
    print_cm(cm, ["negative", "neutral", "positive"])


def run_baseline_embeddings(traindata: Data, testdata: Data, embs: list):
    xtrain = traindata.documents
    ytrain = traindata.labels

    vec = MeanEmbeddingVectorizer(embs)
    classifier = Pipeline([('vec', vec), ('cls', LinearSVC(C=10))])
    classifier.fit(xtrain, ytrain)

    xtest = testdata.documents
    ytest = testdata.labels
    predict = classifier.predict(xtest)
    print(classification_report(ytest, predict))
    cm = confusion_matrix(ytest, predict)
    print_cm(cm, ["negative", "neutral", "positive"])


def run_baseline_tfidf(traindata: Data, testdata: Data):
    print("Running baseline...")

    xtrain = traindata.documents
    ytrain = traindata.labels

    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    classifier = Pipeline(
        [('vec', vec), ('cls', LinearSVC(C=10))])
    classifier.fit(xtrain, ytrain)

    xtest = testdata.documents
    ytest = testdata.labels
    predict = classifier.predict(xtest)
    print(classification_report(ytest, predict))
    cm = confusion_matrix(ytest, predict)
    print_cm(cm, ["negative", "neutral", "positive"])
    return classifier

    # cls_object = classifier.named_steps['cls']
    # vec_object = classifier.named_steps['vec']
    # class_labels = ['negative', 'neutral', 'positive']
    # coef = cls_object.coef_

    # print()
    # feature_names = vec_object.get_feature_names()
    # for i, class_label in enumerate(class_labels):
    #     top10 = np.argsort(cls_object.coef_[i])[-10:]
    #     print("%s: %s" % (class_label,
    #           " ".join(feature_names[j] for j in top10)))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def console(model):
    while True:
        sentence = input("Enter: ")
        if sentence == "EXIT":
            break
        sentence = [sentence.split()]
        print(model.predict(sentence))


if __name__ == "__main__":

    train_conll = Data(
        "../data_files/final_train.conll", format="conll")

    # train_2016 = Data(
    #     "../data_files/2016_spanglish_annotated.json", format="json")

    test = Data("../data_files/final_trial.conll", format="conll")

    # data_set = []

    # for d in [train_2016, test]:
    #     d = Preprocessor.RegFormatter(d)
    #     data_set.append(d)

    # run_baseline_tfidf(data_set[0], data_set[1])

    # from gensim.models import KeyedVectors
    # # emb_en = KeyedVectors.load_word2vec_format(
    # #     "../data_files/wiki.en.align.vec")
    # # emb_es = KeyedVectors.load_word2vec_format(
    # #     "../data_files/wiki.es.align.vec")

    # google = KeyedVectors.load_word2vec_format(
    #     "../data_files/GoogleNews-vectors-negative300.bin", binary=True)

    # # embs = [emb_en, emb_es]
    # embs = [google]

    # print("\nWord embeddings using final_train.conll")
    # run_baseline_embeddings(train_conll, test, embs)

    # print("\nWord embeddings using 2016_spanglish")
    # run_baseline_embeddings(train_2016, test, embs)

    # Enable a console for realtime testing
    # console(model)
