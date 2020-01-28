class Data:
    def __init__(self, path: str = None, format="json"):
        # properties of this class
        self.documents = []
        self.labels = []

        if path != None:
            # lang_tags has the same shape as documents.
            self.documents, self.labels = self.__load_data(path, format)

        self.vectorised = []

    def __load_data(self, path, format):
        print("loading data...")
        with open(path, "r") as file:
            docs = []
            sentiment = []
            sentence = []
            if format == "json":
                import json
                data = json.load(file)
                for d in data:
                    if d["label"] == "sentiment_label":
                        continue

                    docs.append(d["tokens"])
                    sentiment.append(d["label"])

                return docs, sentiment

            for row in file:
                if row == "\n":
                    docs.append(sentence)
                    sentence = []

                else:
                    s = str(row).strip().split('\t')
                    if len(s) >= 3:
                        sentiment.append(s[2])
                        continue

                    sentence.append(s[0])

        return docs, sentiment

    def scramble(self):
        import random
        data = []
        for i in range(len(self.documents)):
            data.append([self.documents[i], self.labels[i]])
        random.shuffle(data)
        self.documents = []
        self.labels = []
        for d in data:
            self.documents.append(d[0])
            self.labels.append(d[1])


class Preprocessor():
    """
    The Preprocessor class is a static class.
    """

    @staticmethod
    def emoji_to_word(data: Data, info_path: str) -> Data:
        import emoji
        import re

        emoji_dic = {}

        with open(info_path, 'r') as file:
            for row in file:
                tokens = row.split("|")
                if tokens[0] not in emoji.UNICODE_EMOJI:
                    continue

                emoji_dic[tokens[0]] = tokens[2]

        new_data = Data()
        for i, doc in enumerate(data.documents):
            new_doc = []
            for token in doc:
                try:
                    text = emoji_dic[token]
                except KeyError:
                    text = token

                new_doc.append(text)
            new_data.documents.append(new_doc)
            new_data.labels.append(data.labels[i])
        return new_data

    @staticmethod
    def split_data(data: Data, split=0.8):
        X = data.documents
        Y = data.labels

        import numpy
        split = numpy.clip(split, 0, 1)

        split_point = int(len(X) * split)

        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]

        Xtest = X[split_point:]
        Ytest = Y[split_point:]

        train = Data()
        train.documents = Xtrain
        train.labels = Ytrain

        test = Data()
        test.documents = Xtest
        test.labels = Ytest

        return train, test

    @staticmethod
    def combine_data(dataA: Data, dataB: Data) -> Data:
        for x, doc in enumerate(dataB.documents):
            dataA.documents.append(doc)
            dataA.labels.append(dataB.labels[x])

        return dataA

    @staticmethod
    def balance_data(data: Data) -> Data:

        positive = data.labels.count("positive")
        negative = data.labels.count("negative")
        neutral = data.labels.count("neutral")

        max_len = min([positive, negative, neutral])

        blanced_data = Data()

        count = {}

        for x, doc in enumerate(data.documents):

            if data.labels[x] not in count:
                count[data.labels[x]] = 1
            else:
                if count[data.labels[x]] <= max_len:
                    count[data.labels[x]] += 1
                else:
                    continue

            blanced_data.documents.append(doc)
            blanced_data.labels.append(data.labels[x])

        return blanced_data

    @staticmethod
    def remove_stopwords(data: Data, language: str) -> Data:
        from nltk.corpus import stopwords

        wordlist = set(stopwords.words(language))

        docs = []
        for i, doc in enumerate(data.documents):
            tokens = []
            for x, token in enumerate(doc):
                if token.lower() not in wordlist:
                    tokens.append(token)

            if len(tokens) == 0:
                tokens.append("Nani!?")
                print("Nani?!")

            docs.append(tokens)

        data.documents = docs
        return data

    @staticmethod
    def remove_punctuations(data: Data) -> Data:
        puctuations = list(',?/.()[];:\'\"')
        docs = []
        for i, doc in enumerate(data.documents):
            tokens = []
            for x, token in enumerate(doc):
                if token not in puctuations:
                    tokens.append(token)

            docs.append(tokens)

        data.documents = docs
        return data

    @staticmethod
    def remove_emoji(data: Data) -> Data:
        print("Removing emojis...")
        from emoji import UNICODE_EMOJI, demojize

        newdata = Data()
        for x, doc in enumerate(data.documents):
            for token in doc:
                if token in UNICODE_EMOJI:
                    doc.remove(token)
                    continue

                for c in str(token):
                    if c in UNICODE_EMOJI:
                        doc.remove(token)
                        break

            newdata.documents.append(doc)
            newdata.labels.append(data.labels[x])

        return newdata

    @staticmethod
    def RegFormatter(data: Data, pattern="\w+") -> Data:
        from nltk.tokenize import RegexpTokenizer

        new_data = Data()

        for i, doc in enumerate(data.documents):
            sent = ""
            for token in doc:
                sent += str(token) + " "

            tokenizer = RegexpTokenizer(pattern)
            tokens = tokenizer.tokenize(sent)

            new_data.documents.append(tokens)
            new_data.labels.append(data.labels[i])

        return new_data

    @staticmethod
    def remove_dup(data: Data) -> Data:
        print("Removing duplication...")
        import pandas

        dic = {"sentences": [], "index": []}

        for i, doc in enumerate(data.documents):

            dic["sentences"].append(hash(tuple(doc)))
            dic["index"].append(i)

        df = pandas.DataFrame(data=dic)
        df.drop_duplicates(subset=['sentences'], keep=False)

        new_data = Data()
        for item in df["index"]:
            new_data.documents.append(data.documents[item])
            new_data.labels.append(data.labels[item])

        return new_data


class Explorer:
    @staticmethod
    def count_docs(data: Data):
        length = str(len(data.documents))
        print("There are a total of " + length + " documents.")

        print("There are " + str(data.labels.count("positive")) +
              " positive documents")
        print("There are " + str(data.labels.count("negative")) +
              " negative documents")

        print("There are " + str(data.labels.count("neutral")) +
              " neutral documents")
        return


# For debugging purposes
if __name__ == "__main__":
    # data = Data("../../data_files/2016_spanglish_annotated.json")
    data = Data("./final_train.conll", format="conll")
    Explorer.count_docs(data)
    # data = Preprocessor.combine_data(data, data)
    # Explorer.count_docs(data)
