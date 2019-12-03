from data.data_manager import Data, Preprocessor
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *

def run_baseline(traindata:Data, testdata:Data):
    xtrain = traindata.documents
    ytrain = traindata.labels

    vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer= lambda x: x)
    classifier = Pipeline([('vec', vec), ('cls', SVC(C= 0.6, gamma="auto"))])
    classifier.fit(xtrain, ytrain)

    xtest = testdata.documents
    ytest = testdata.labels
    predict = classifier.predict(xtest)
    print(classification_report(ytest, predict))

if __name__ == "__main__":
    data = Data("../data_files/train_conll_spanglish.txt")
    # data = Preprocessor.emoji_to_word(data)
    data.scramble()

    traindata, testdata = Preprocessor.split_data(data)
    run_baseline(traindata, testdata)