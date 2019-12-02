class Data:
    def __init__(self, path=None):
        # properties of this class
        self.documents = []
        self.labels = []
        self.lang_tags = []

        # lang_tags has the same shape as documents.
        self.documents, self.labels, self.lang_tags = self.load_data(path)

    def load_data(self, path):
        print("loading data...", end="")
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

class Preprocessor():
    def __init__(self):
        pass
    
    @staticmethod
    def process():
        print("hello")
        pass



# For debugging purposes
if __name__ == "__main__": 
    data = Data("../../data_files/spanglish_trial.txt")
    pass
