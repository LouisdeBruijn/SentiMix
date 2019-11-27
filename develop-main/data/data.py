class Data:

    def __init__(self, path=None):

        # properties of this class
        self.documents = []
        self.labels = []
        self.lang_tags = []

        # lang_tags has the same shape as documents.

        self.documents, self.labels, self.lang_tags = self.load_data(path)
        pass

    def load_data(self, path):

        print("loading data...", end="")
        with open(path, "r") as file:
            docs = []
            langs = []
            sentence = []
            lang = []
            sentiment = []

            for row in file:
                if row == "\n":
                    docs.append(sentence)
                    langs.append(lang)
                    sentence = []
                else:
                    s = str(row).strip().split('\t')
                    if len(s) >= 3:
                        if len(s) == 3:
                            sentiment.append(s[2])
                        continue

                    sentence.append(s[0])
                    lang.append(s[1])

        return docs, sentiment, langs

if __name__ == "__main__": 
    
    pass
