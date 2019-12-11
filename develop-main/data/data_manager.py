class Data:
    def __init__(self, path: str = None, format="json"):
        # properties of this class
        self.documents = []
        self.labels = []

        if path != None:
            # lang_tags has the same shape as documents.
            self.documents, self.labels = self.__load_data(
                path, format)

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
    def load_embeddings(data: Data, embeddings: list, unkown_vectors: list = None, vector_length: int = 300) -> Data:
        import random
        docs = data.documents
        for doc in docs:
            vectors = []
            for token in doc:
                
                if unkown_vectors == None:
                    vec = [random.uniform(-1.0, 1.0) for n in range(vector_length)]
                else:
                    vec = unkown_vectors
                
                for emb in embeddings:
                    try:
                        emb[token]
                    except KeyError:
                        continue
                
                vectors.append(vec)

            data.vectorised.append(vectors)

        return data

    @staticmethod
    def emoji_to_word(data: Data) -> Data:
        import emoji
        import re

        docs = data.documents
        for i, doc in enumerate(docs):
            processed = []
            for x, token in enumerate(doc):
                token = emoji.demojize(token)
                token = re.sub(r'[^\w\s]', '', token)
                token = token.split("_")
                processed.extend(token)

            docs[i] = processed

        data.documents = docs

        return data

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
        
        positive = 0
        negative = 0
        neutral = 0
        
        for label in data.labels:
            if label == "positive":
                positive += 1
            elif label == "negative":
                negative += 1
            elif label == "neutral":
                neutral += 1

        max_len = min([positive, negative, neutral])
        
        data.scramble()

        blanced_data = Data()

        pos_count = 0
        neg_count = 0
        neu_count = 0

        for x, doc in enumerate(data.documents):
            
            if data.labels[x] == "positive":
                if pos_count >= max_len:
                    continue
                else:
                    pos_count += 1

            if data.labels[x] == "negative":
                if neg_count >= max_len:
                    continue
                else:
                    neg_count += 1

            if data.labels[x] == "neutral":
                if neu_count >= max_len:
                    continue
                else:
                    neu_count += 1    

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

    embs = [en_embs, es_embs]

    # Because Preprocessor is a static class,
    # we do not need to instantiate an object.
    data = Preprocessor.load_embeddings(data, embs)

    # Because Prepocessor takes type Data as argument and returns type Data as value,
    # the process is very straight forward.
    print(data.vectorised[0])
