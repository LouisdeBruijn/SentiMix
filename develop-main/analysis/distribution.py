from emoji import UNICODE_EMOJI
import operator
from collections import defaultdict, Counter
import numpy as np
import json
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import json
import itertools
plt.style.use('ggplot')


class Data:
    """Reads data from trial data files"""

    path = ""
    labels = []
    documents = []

    def __init__(self, path):
        self.path = path
        self.__load()

    def __load(self):
        print("loading data...\n")
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

    def __create_new_txt(self):
        print("creating new txt file...\n")
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


class NewData:
    """Reads data from new tweets file"""

    path = ""
    labels = []
    documents = []

    def __init__(self, path):
        self.path = path
        self.__load()

    def __load(self):
        print("loading data...\n")
        with open(self.path, "r") as file:
            docs = []
            sentiment = []
            for tweet in file:
                docs.append(tweet)
                sentiment.append('')

        self.documents = docs
        self.labels = sentiment


def new_emoji_labels():
    """
    Converts old labels to new labels based on most_informative features
    """
    new_labels = {}
    with open('../dist/emoji_labels.txt', 'r') as new_emoji_labels:
        next(new_emoji_labels)
        for line in new_emoji_labels:
            line = line.rstrip().split('|')
            new_labels[line[0]] = line[1]

    return new_labels


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
    nr_emojis_in_tweet = {}
    new_emoji = defaultdict(list)

    # new labels based on most-informative features (Gaetana)
    new_labels = new_emoji_labels()

    for doc, label in zip(data.documents, data.labels):
        # convert labels to new label
        emojis_present = False
        for token in doc:
            for char in token:
                if char in UNICODE_EMOJI:
                    '''this is for finding the labels per emoticon'''
                    # converts old label to new label based on most-inf features
                    new_emoji[char].append(label)

                    emojis_present = True
                    '''distribution of total occurences of emojis'''
                    # if char is already in dictionary, add +1 to count
                    # if char is not in dictionary, add it with count 1
                    emojis[char] = emojis.get(char, 0)+1

        if emojis_present:
            '''distribution of emojis present in tweets'''
            emoji_in_doc += 1

    # converting labels to counts and sort it high-low
    sorted_emoji = conv_labels_counts(new_emoji)

    with open('../dist/emoji_informativity.txt', 'a') as out_file:
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

            # new labels based on most-informative features (Gaetana)
            new_labels = new_emoji_labels()
            new_label = new_labels[labels[0][0]]

            items = [char, labels[0][0], new_label, str(tot), tup[1]]

            # converting Counter to string
            string = ''
            for k, v in items[4].items():
                string += k +':' + str(v) + ' '
            items[4] = string

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


def emoji_dic(emoji_informativity, new_labels):
    """
    Returns dictionary key-value pair of emoji-label

    :param str emoji_informativity: path for output emoji informativity file
    :param bool new_labels: if True we use new_labels from most inf features

    :rtype dict
    :returns: dictionary with emojis as keys and labels as value
    """
    with open("../dist/emoji_informativity.txt") as emoji_file:
        next(emoji_file) # skip filename
        next(emoji_file) # skip headers
        emojis = {}
        for line in emoji_file:
            line = line.rstrip().split('|')
            if new_labels:
                emojis[line[0]] = line[2]
            else:
                emojis[line[0]] = line[1]

    return emojis


def conv_labels_counts(new_emoji):
    """finds labels per emoticon

    :param dict new_emoji: dictionary with emojis as keys
    and a list of labels as values

    :rtype tup
    :return: sorted emojis by count of labels and their frequencys
    """

    # converting the list of labels to counts
    counted_emoji = {}
    for char, label in new_emoji.items():
        cnt = Counter(label)
        freq = sum(cnt.values())
        if freq > 1: # we're not considering emojis that only occur once
            counted_emoji[char] = (freq, cnt)

    # sorting it highest value first
    sorted_emoji = sorted(counted_emoji.items(),
        key=lambda elem: elem[1][0],
        reverse=True)

    return sorted_emoji


def label_new_tweets(new_data, filename, emojis, threshold):
    """labels newly found tweets based on the most informative emoji labels

    :param class new_data: NewData class with tweets
    :param str filename: name of the trial data file
    :param emojis dict: most informative emojis and their labels
    :param int threshold: threshold in percentage how much 1st and 2nd
        most found emoji in tweet should be in order to be labelled

    :rtype txt
    :return: tab-delimited text file with labelled tweets
    """

    emoji_in_doc = 0
    for idx, doc in enumerate(new_data.documents):
        emojis_present = False
        found_emojis = {}

        for char in doc:
            if char in UNICODE_EMOJI:
                emojis_present = True

                if char in emojis.keys():
                    # char is found in most_informative emojis
                    label = emojis[char]
                    found_emojis[label] = found_emojis.get(label, 0)+1

        if emojis_present:
            '''distribution of emojis present in tweets'''
            emoji_in_doc += 1

        if found_emojis:
            '''emojis were found in this tweet'''

            if len(found_emojis) > 1:
                '''there are multiple labels found for the emojis in one tweet'''

                freq = sum(found_emojis.values())
                for label, count in found_emojis.items():
                    # convert counts to percentages
                    found_emojis[label] = round((count/freq)*100, 2)

                # sorting it highest value first
                # and keeping only the highest and 2nd highest value
                sort_temp_label = sorted(found_emojis.items(),
                    key=operator.itemgetter(1),
                    reverse=True)[:2]

                # find difference between 1st and 2nd most-used emoji in tweet
                diff = abs(np.diff([n for _, n in sort_temp_label[:2]]))
                if diff[0] <= threshold:
                    continue

                # returning percentages to original counts of the 1st highest label
                count = int((sort_temp_label[0][1]/100) * freq)
                label = sort_temp_label[0][0]
                found_emojis = {}
                found_emojis[label] = count

            # assign the label to this tweet
            label = list(found_emojis.keys())[0]
            new_data.labels[idx] = label

    distr = {} # distribution of sentiment labels in newly labelled tweets
    # write to output txt file
    with open("../../data_files/2016_spanglish_annotated.tweets", "a") as tweet_output:
        tweet_output.write("sentiment_label\ttweet_text\n")
        for doc, label in zip(new_data.documents, new_data.labels):
            if label:
                tweet_output.write("{0}\t{1}".format(label, doc))
                distr[label] = distr.get(label, 0)+1

    # some descriptive statistics (also in /dist)
    emojis_occur = 'Emojis occur in {0}% of the documents: {1}'.format( round((emoji_in_doc/len(new_data.documents))*100, 2), emoji_in_doc)
    tweets_labelled = len([label for label in new_data.labels if label != ''])
    labelled = 'Tweets that could be labelled: {0}'.format(tweets_labelled)
    print(emojis_occur)
    print(labelled)
    print(distr)
    with open("../dist/2016_spanglish_annotated.distr", "w") as out_file:
        out_file.write(emojis_occur + '\n')
        out_file.write(labelled + '\n')
        out_file.write(json.dumps(distr) + '\n')


def tokenize_tweets():
    """
    Tokenize both conll_spanish and new 2016 annotated tweets and write to JSON file
    """
    tknzr = TweetTokenizer()

    # old conll spanglish training data
    datafile = "../../data_files/train_conll_spanglish.txt"
    data = Data(datafile)

    conll_json = []
    print("Tokenizing conll_spanglish data...\n")
    for idx, (label, tokens) in enumerate(zip(data.labels, data.documents)):
        idx += 1
        tweet = " ".join(tokens)
        tokenized = tknzr.tokenize(tweet)
        emojis = emoji_dic("../dist/emoji_informativity.txt", True)
        emoji_conv = convert_emojis_to_labels(tokenized, emojis)
        conll_json.append({'id': idx, 'label': label, 'tokens': emoji_conv})

    with open("../../data_files/conv_train_conll_spanglish.json", "w") as conll_json_file:
        json.dump(conll_json, conll_json_file)

    # new 2016 annotated tweets
    tweets_2016_json = []
    print("Tokenizing new 2016 annotated tweets data...\n")
    with open("../../data_files/2016_spanglish_annotated.tweets", "r") as tweets_2016_file:
        next(tweets_2016_file)
        for line in tweets_2016_file:
            idx += 1
            label, doc = line.rstrip().split('\t')
            tokenized = tknzr.tokenize(doc)
            emojis = emoji_dic("../dist/emoji_informativity.txt", True)
            emoji_conv = convert_emojis_to_labels(tokenized, emojis)
            tweets_2016_json.append({'id': idx, 'label': label, 'tokens': tokenized})

    with open("../../data_files/conv_2016_spanglish_annotated.json", "w") as tweets_2016_json_file:
        json.dump(tweets_2016_json, tweets_2016_json_file)

    print('Tokenization for both old and new data done\n')


def convert_emojis_to_labels(li_tokens, emojis):
    """
    Converts an emoji to the labels from the most informative features (Gaetana)

    :param list li_tokens: tokens that contain the emojis to convert
    :param dict emojis: dictionary with emojis as keys
        and labels to convert emojis to as the values

    :rtype: list
    :return: list w/ tokens and converted emojis
    """
    for idx, token in enumerate(li_tokens):
        if token in UNICODE_EMOJI and token in emojis.keys():
            new_token = emojis[token]
            li_tokens[idx] = new_token

    return li_tokens


def main():

    # TODO: we still need to trim the amount of informative emojis!!

    tokenize_tweets()

    exit()
    """Labeling new tweets"""
    new_tweets_file = "../../data_files/2016.tweets"
    new_data = NewData(new_tweets_file)
    file = new_tweets_file.split('/')[-1] # filename
    emojis = emoji_dic("../dist/emoji_informativity.txt", False)
    label_new_tweets(new_data, file, emojis, 20)

    """Creating most informative emojis txt file"""
    # Loading the data
    datafile = "../../data_files/train_conll_spanglish.txt"
    file = datafile.split('/')[-1] # filename
    data = Data(datafile)
    most_inf_emojis(data, file, 20)



if __name__ == '__main__':
    main()
