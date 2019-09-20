import os 

dt_path = "./train_conll_spanglish.txt"
class NGram:
    word = ""
    lang = ""

#read the data from codalab
def read_data(path):
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
                #this adds the labels
                labels.append(l[len(l)-1])
            else:
                w = line.strip().split()
                gram = NGram()
                gram.word = w[0]
                gram.lang = w[1]
                #only assigning the actual word for now.
                sentence.append(gram.word)
        else:
            start = True
            docs.append(sentence)
            continue

    return docs, labels


X,Y = read_data(dt_path)
print(Y)