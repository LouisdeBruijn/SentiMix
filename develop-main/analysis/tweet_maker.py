# File name: 
# Description: create a files with tweets, language tags and sentiments separated
# Author: Marion Bartl
# Date: 20-11-19

from collections import defaultdict


def open_file(file_path):
    '''returns tweets in a dictionary with a tuple of index and sentiment as [key] and tuples of tokens and language as [value]'''

    # extract the tweets from the .txt files and put in a dictionary
    tweets = defaultdict(list)
    with open(file_path, 'r') as spanglish_trial:
        for line in spanglish_trial:
            line = line.strip().split('\t')

            if line[0] == 'meta':
                if len(line) == 3:
                    idx = line[1]
                    sentiment = line[2]
                else:
                    print('sentiment missing')
                    pass
            elif '' not in line:
                tweets[(idx, sentiment)].append((line[0], line[1]))
    return tweets


if __name__ == '__main__':
    filename = 'train_conll_spanglish.txt'
    tweets = open_file(filename)

    texts = []
    tags = []
    sentiments = []

    for k, tweet in tweets.items():
        sentiments.append(k[1])
        words = [w for (w,l) in tweet]
        texts.append(words)
        lang_tags = [l for (w,l) in tweet]
        tags.append(lang_tags)

    with open('raw_tweets.txt', 'w') as raw_file:
        for tweet in texts:
            sent = ' '.join(tweet)+'\n'
            raw_file.write(sent)
    with open('lang_tags.txt', 'w') as tag_file:
        for tweet in tags:
            sent = ' '.join(tweet)+'\n'
            tag_file.write(sent)

    with open('sentiments.txt', 'w') as sent_file:
        for s in sentiments:
            sent_file.write(s+'\n')
