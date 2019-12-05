# File name: norm_exam.py
# Description: Inspect if the normalization of tweets was successful (in terms of no. of words that were lost)
# Author: Marion Bartl
# Date: 20-11-19

def read_file(filename):
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 1:
                tweets.append(line[0])
            else:
                tweets.append(line)
    return tweets

if __name__ == '__main__':

    norm_file = "norm_es.txt"
    tag_file = "lang_tags.txt"
    sent_file = 'sentiments.txt'

    tweets = read_file(norm_file)
    lang_tags = read_file(tag_file)
    sentiments = read_file(sent_file)

    print(len(tweets), len(lang_tags), len(sentiments))

    for tweet, l_tags in zip(tweets, lang_tags):
        if len(tweet) != len(l_tags):
            print(tweet)
            print(l_tags)
    else:
        print('There are no tweets that have a different length after conversion.')


