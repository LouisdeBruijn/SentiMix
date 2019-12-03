from emoji import UNICODE_EMOJI
import operator
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Data:
    """Reads data from data files"""

    path = ""
    labels = []
    documents = []

    def __init__(self, path):
        self.path = path
        self.__load()

    def __load(self):
        print("loading data...", end="")
        with open(self.path, "r") as file:
            docs = []
            sentence = []
            sentiment = []
            for row in file:
                if row == "\n":
                    docs.append(sentence)
                    sentence = []
                else:
                    s = str(row).strip().split('\t')
                    if len(s) >= 3:
                        if len(s) == 3:
                            sentiment.append(s[2])
                        continue

                    sentence.append(s[0])

        self.documents = docs
        self.labels = sentiment



def most_inf_emojis(data, filename, threshold):
    """Creates a txt file with most informative emojis

    :param class data: Data class with trial data
    :param str filename: name of the trial data file
    :param int threshold: minimal difference in percentage between
    1st and 2nd most found label for each emoji

    :rtype txt
    :return: pipe-delimited text file with most informative emojis
    """

    emoji_in_doc = 0
    emojis = {}
    new_emoji = defaultdict(list)
    for doc, label in zip(data.documents, data.labels):
        for token in doc:
            emojis_present = False
            for char in token:
                if char in UNICODE_EMOJI:
                    '''this is for finding the labels per emoticon'''
                    new_emoji[char].append(label)

                    emojis_present = True
                    '''distribution of total occurences of emojis'''
                    # if char is already in dictionary, add +1 to count
                    # if char is not in dictionary, add it with count 1
                    emojis[char] = emojis.get(char, 0)+1
            if emojis_present:
                '''distribution of emojis present in tweets'''
                emoji_in_doc += 1

    ''' for finding the labels per emotion'''
    # converting the list of labels to counts
    counted_emoji = {}
    for char, label in new_emoji.items():
        cnt = Counter(label)
        freq = sum(cnt.values())
        if freq > 1: # we're not considering emojis that only occur once
            counted_emoji[char] = (freq, cnt)

    # sorting it
    sorted_emoji = sorted(counted_emoji.items(), key=lambda elem: elem[1][0], reverse=True)

    with open('emoji_informativity.txt', 'a') as out_file:
        out_file.write(filename + '\n')
        out_file.write('emoji|chosen label|nr of occurences in tweets|label distribution' + '\n')

        # convert counts to percentages
        for char, tup in sorted_emoji:
            tot = tup[0]
            labels = tup[1].most_common(2)
            perc = []
            for label, count in labels:
                count = round((count/tot)*100, 2)
                perc.append((label, count))

            '''set threshold of at least % difference between
            1st most common and 2nd most common label'''
            if len(perc) == 1: # there is only one label
                pass
            else:
                diff = abs(np.diff([n for _, n in perc]))
                if diff[0] <= threshold:
                    continue

            items = [char, labels[0][0], str(tot), tup[1]]
            # converting Counter to string
            string = ''
            for k, v in items[3].items():
                string += k +':' + str(v) + ' '
            items[3] = string

            # write to file
            # print('|'.join(items) + '\n')
            out_file.write('|'.join(items) + '\n')

    # descriptive statistics and visuals
    descriptive_stats(data, filename, emojis, emoji_in_doc)


def descriptive_stats(data, filename, emojis, emoji_in_doc):
    """Returns descriptive statistics for trial data

    :param class data: Data class with trial data
    :param str filename: name of the trial data file
    :param dict emojis: all emojis occuring in the tweets
    :param int emoji_in_doc: count of the occurences of emojis in tweets

    :rtype str
    :return: printed descriptive statistics and graphical visualisations
    """
    print('')
    print('Total number of documents in {0}: {1}'.format(filename, len(data.documents)))
    print('Total number of emojis in {0}: {1}'.format(filename, sum(emojis.values())))
    print('Emojis occur in {0}% of the documents: {1}'.format( round((emoji_in_doc/len(data.documents))*100, 2), emoji_in_doc))

    # sort emojis according to count
    sorted_emojis = sorted(emojis.items(), key=operator.itemgetter(1), reverse=True)
    print('')
    print('Find the emoji distribution below.')
    print(sorted_emojis)

    # top 10 for visualisation
    labels, sizes = zip(*sorted_emojis[:10])
    print('')
    print('labels for visualisations: {0}'.format(labels))

    # bar plot
    x_pos = [i for i, _ in enumerate(labels)]
    plt.bar(x_pos, sizes, color='blue')
    plt.xlabel("Emoji labels")
    plt.ylabel("Total counts")
    plt.title("Distribution of emoji counts")
    plt.xticks(x_pos, labels)
    plt.show()

    # pie chart
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=labels,
    autopct='%1.1f%%', shadow=False, startangle=140)
    plt.axis('equal')
    plt.show()


def main():

    with open("../../data_files/2016.tweets") as tweets:
        for line in tweets:
            print(line)
            exit()


    # Loading the data
    datafile = "../../data_files/train_conll_spanglish.txt"
    file = datafile.split('/')[1][:-4] # filename
    data = Data(datafile)

    # most_inf_emojis(data, file, 20)






if __name__ == '__main__':
    main()
