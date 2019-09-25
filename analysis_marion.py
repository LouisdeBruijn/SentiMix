#!/usr/bin/env python3
# File name: analysis.py
# Description: Preliminary data-analysis for Shared Task
# Authors: Louis de Bruijn
# Date: 21-09-2019
import operator
from collections import Counter, defaultdict
import sys

import nltk
from nltk.probability import FreqDist



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


def lang_dist(tweets):
    '''returns distribution of the languages of the tokens'''

    cnt = Counter()
    distribution_lists = defaultdict(list)
    for (idx, sentiment), tup in tweets.items():

        # count language tags
        for token, lang in tup:
            cnt[lang] += 1

    return cnt


def sent_dist(tweets):
    '''a function to check how sentiment is di    for sent, count in sent_counts.items():
stributed across the tweets'''

    # count sentiments
    cnt = Counter()
    for idx, sentiment in tweets.keys():
        cnt[sentiment] += 1

    return cnt


def tweet_dist(counts, tweets):
    '''distribution of the counts per language normalised by their total count'''

    dic = {}
    for (idx, sentiment), tup in tweets.items():
        cnt = Counter()
        for token, lang in tup:
            cnt[lang] += 1
        
        distr = set()
        for lang, value in cnt.items():
            perc = round(value/sum(cnt.values()), 1)
            distr.add((lang, perc))

        tup = tuple(distr)

        dic[(idx, sentiment)] = tup

    counter = {}
    for tup, distr in dic.items():
        # print(tup, distr)
        counter[distr] = counter.get(distr, 0) + 1


    print(counter)
    # counting distributions
    # mydict = {}
    # for word in words:
    #     mydict[word]=mydict.get(word ,0)+1

def main():

    lang1 = 'English'
    lang2 = 'Spanish'

    file_path = 'spanglish_trial.txt'
    tweets = open_file(sys.argv[1])

    # get the distribution of sentiments across tweets
    # sent_counts = sent_dist(tweets)
    # print("\nDistribution of sentiment across tweets\n")
    # for sent, count in sent_counts.items():
    #     print(sent, count)

    # counts

    counts = lang_dist(tweets)

    for key, value in counts.items():
        if key == 'lang1':
            print(lang1, value)
        elif key == 'lang2':
            print(lang2, value)
        else: 
            print(key ,value)

    tweet_dist(counts, tweets)


    exit()
        # turn language counts by word into fractions
        # for lang, value in cnt.items():
        #     cnt[lang] = value/sum(cnt.values())

        # sort languages in a tweet by occurence
        # lang_freq = sorted(cnt.items(), key=operator.itemgetter(1), reverse = True)

        # if lang_freq[0][1] >= 0.75:
        #     distribution_lists[sentiment].append([lang_freq[0][0]])
        # if lang_freq[0][1] >= 0.5 and len(lang_freq) >= 2:
        #     distribution_lists[sentiment].append([lang_freq[0][0], lang_freq[1][0]])
        # else:
        #     distribution_lists[sentiment].append([lang for lang, freq in lang_freq if freq > 0.3])


    # dictionary with sentiment as keys and COUNTS of languages as values
    distribution_counts = {}
    for sentiment, languages in distribution_lists.items():
        distr = Counter()
        for val in languages:
            distr[tuple(val)] += 1
        inside_distr = sorted(distr.items(), key=operator.itemgetter(1), reverse = True)
        distribution_counts[sentiment] = dict(inside_distr)


    return distribution_counts, distribution_lists


    # print("\nDistribution of majority languages across sentiment")
    # for sentiment, value in counts.items():
    #     print('\n'+sentiment)
    #     for k, v in value.items():
    #         langs = []
    #         for lang in k:
    #             if lang == 'lang1':
    #                 langs.append(lang1)
    #             elif lang == 'lang2':
    #                 langs.append(lang2)
    #             else:
    #                 langs.append(lang)
    #         print(langs, v)

if __name__ == '__main__':
    main()