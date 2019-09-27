
class Data:
    def __init__(self, train_set_path, test_set_path=None, split=0.5, combine=False):
        self.x_train, self.y_train, self.x_test, self.y_test = self.__get_data(
            train_set_path, test_set_path, split, combine)

    # This output data into 4 seperate vars so that you don't need to get them by doing things like data.x_trian.
    def output_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    # read the data from codalab
    def __read_data(self, path):
        """
        Opens Conll data txts and outputs the documents and labels
        """
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
                    # this adds the labels
                    labels.append(l[len(l)-1])
                else:
                    w = line.strip().split()

                    if len(w) < 2:
                        continue

                    # only assigning the actual word for now.
                    # This can be modified later for implementing language labels
                    sentence.append(w[0])
            else:
                start = True
                docs.append(sentence)
                continue
        return docs, labels

    def __get_data(self, dataset_path, testdata_path=None, split=0.5, combine=False):
        """
        Automatically output Xtrain Ytrain Xtest and Ytest from the data extracted from read_data()
        """
        X, Y = self.__read_data(
            dataset_path)  # This outputs documents and labels of the training data
        split_point = int(split * len(X))

        if testdata_path == None:
            Xtrain = X[:split_point]
            Ytrain = Y[:split_point]

            Xtest = X[split_point:]
            Ytest = Y[split_point:]
        else:
            Xtrain = X
            Ytrain = Y
            # This outputs documents and labels of the testing data
            Xtest, Ytest = self.__read_data(testdata_path)
            if combine:
                Xtrain = X + Xtest
                Ytrain = Y + Ytest

        return Xtrain, Ytrain, Xtest, Ytest
