from data.data_manager import Data, Preprocessor
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *

import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        # X should be data.vectorised
        vec = []
        for vectors in X:
            import numpy
            v = numpy.array(vectors)
            v = numpy.mean(v, axis=0)
            vec.append(v)

        return vec

def run_baseline_embeddings(traindata:Data, testdata:Data):
    xtrain = traindata.vectorised
    ytrain = traindata.labels

    vec = MeanEmbeddingVectorizer()
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])
    classifier.fit(xtrain, ytrain)

    xtest = testdata.vectorised
    ytest = testdata.labels
    predict = classifier.predict(xtest)
    print(classification_report(ytest, predict))
    cm = confusion_matrix(ytest, predict)
    print_cm(cm, ["negative", "neutral", "positive"])

def run_baseline_tfidf(traindata:Data, testdata:Data):
    xtrain = traindata.documents
    ytrain = traindata.labels

    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer= lambda x: x)
    classifier = Pipeline([('vec', vec), ('cls', RandomForestClassifier(n_estimators=1000))])
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
    
    # data = Data("../data_files/train_conll_spanglish.txt", format="conll")
    data = Data("../data_files/2016_spanglish_annotated.json", format="json")

    # data = Preprocessor.combine_data(data, new)
    
    data = Preprocessor.balance_data(data)
    data.scramble()

    traindata, testdata = Preprocessor.split_data(data, 0.8)

    model = run_baseline_tfidf(traindata, testdata)
    console(model)