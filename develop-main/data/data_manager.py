class Data:
    def __init__(self, path: str = None):
        # properties of this class
        self.documents = []
        self.labels = []
        self.lang_tags = []

        if path != None:
            # lang_tags has the same shape as documents.
            self.documents, self.labels, self.lang_tags = self.__load_data(
                path)

        self.vectorised = []

    def __load_data(self, path):
        print("loading data...")
        with open(path, "r") as file:
            docs = []
            langs = []
            sentiment = []

            sentence = []
            lang = []

            for row in file:
                if row == "\n":
                    docs.append(sentence)
                    langs.append(lang)
                    sentence = []
                    lang = []
                else:
                    s = str(row).strip().split('\t')
                    if len(s) >= 3:
                        sentiment.append(s[2])
                        continue

                    sentence.append(s[0])
                    lang.append(s[1])

        return docs, sentiment, langs

    def scramble(self):
        import random
        data = []
        for i in range(len(self.documents)):
            data.append([self.documents[i], self.labels[i], self.lang_tags[i]])
        random.shuffle(data)
        self.documents = []
        self.labels = []
        self.lang_tags = []
        for d in data:
            self.documents.append(d[0])
            self.labels.append(d[1])
            self.lang_tags.append(d[2])


class Preprocessor():
    """
    The Preprocessor class is a static class.
    """

    @staticmethod
    def load_embeddings(data: Data, embeddings: dict, unkown_vectors: list = None, vector_length: int = 300) -> Data:
        import random
        docs = data.documents
        for i, doc in enumerate(docs):
            vectors = []
            for x, token in enumerate(doc):
                try:
                    vec = embeddings[data.lang_tags[i][x]][token]
                    vector_length = len(vec)
                except KeyError:
                    if unkown_vectors == None:
                        # Providing an UNK vector for the embedings.
                        # If there isn't any, a random one will be generated.
                        # It is best to provide one from the embeddings
                        vec = [random.uniform(-1.0, 1.0)
                               for n in range(vector_length)]
                    else:
                        vec = unkown_vectors

                vectors.append(vec)

            data.vectorised.append(vectors)

        return data

    @staticmethod
    def emoji_to_word(data: Data) -> Data:
        import emoji
        import re

        docs = data.documents
        langs = data.lang_tags
        for i, doc in enumerate(docs):
            processed = []
            new_langs = []
            for x, token in enumerate(doc):
                token = emoji.demojize(token)
                token = re.sub(r'[^\w\s]', '', token)
                token = token.split("_")
                for word in token:
                    new_langs.append(langs[i][x])

                processed.extend(token)

            docs[i] = processed
            langs[i] = new_langs

        data.documents = docs
        data.lang_tags = langs

        return data

    @staticmethod
    def split_data(data: Data, split=0.8):
        X = data.documents
        Y = data.labels
        Z = data.lang_tags

        import numpy
        split = numpy.clip(split, 0, 1)

        split_point = int(len(X) * split)

        Xtrain = X[:split_point]
        Ytrain = Y[:split_point]
        Ztrain = Z[:split_point]

        Xtest = X[split_point:]
        Ytest = Y[split_point:]
        Ztest = Z[split_point:]

        train = Data()
        train.documents = Xtrain
        train.labels = Ytrain
        train.lang_tags = Ztrain

        test = Data()
        test.documents = Xtest
        test.labels = Ytest
        test.lang_tags = Ztest

        return train, test

    @staticmethod
    def normalize(data: Data) -> Data:
        # Do we need this?
        pass

    @staticmethod
    def remove_stopwords(data: Data, language: str) -> Data:
        from nltk.corpus import stopwords

        wordlist = set(stopwords.words(language))
        
        docs = []
        langs = []
        for i, doc in enumerate(data.documents):
            tokens = []
            newlangs = []
            for x, token in enumerate(doc):
                if token.lower() not in wordlist:
                    tokens.append(token)
                    newlangs.append(data.lang_tags[i][x])
            
            if len(tokens) == 0:
                tokens.append("Nani!?")
                print("Nani?!")

            docs.append(tokens)
            langs.append(newlangs)
        
        data.documents = docs
        data.lang_tags = langs
        return data


# For debugging purposes
if __name__ == "__main__":
    data = Data("../../data_files/spanglish_trial.txt")

    # Example for loading embedings
    from gensim.models import KeyedVectors

    print("loading english embeddings...")
    en_embs = KeyedVectors.load_word2vec_format(
        "../../data_files/wiki.en.align.vec", limit=10000)

    print("loading spanish embeddings...")
    es_embs = KeyedVectors.load_word2vec_format(
        "../../data_files/wiki.es.align.vec", limit=10000)

    embedding_dict = {
        "lang1": en_embs,
        "lang2": es_embs
    }

    # Because Preprocessor is a static class,
    # we do not need to instantiate an object.
    data = Preprocessor.load_embeddings(data, embedding_dict)

    # Because Prepocessor takes type Data as argument and returns type Data as value,
    # the process is very straight forward.
    print(data.vectorised[0])
