#!/usr/bin/env python3
# File name: analysis.py
# Description: Preliminary data-analysis for Shared Task
# Authors: Louis de Bruijn
# Date: 21-09-2019


from collections import Counter, defaultdict


def open_file(file_path):
    '''returns tweets in a dictionary with a tuple of index and sentiment as [key] and tuples of tokens and language as [value]'''

    # extract the tweets from the .txt files and put in a dictionary
    tweets = defaultdict(list)
    with open('../Sentimix/spanglish_trial.txt', 'r') as spanglish_trial:
        for line in spanglish_trial:
            line = line.strip().split('\t')

            if line[0] == 'meta':
                idx = line[1]
                sentiment = line[2]
            elif '' not in line:
                tweets[(idx, sentiment)].append((line[0], line[1]))

    return tweets


def extract_distribution(tweets):
    '''returns distribution in counts and lists of languages per sentiment '''

    # let's take a look at the distribution of languages in tweets per sentiment: pos, neg, neutral
    # a dictionary with sentiment as keys and a LIST of the language with most tokens in the tweet
    distribution_lists = defaultdict(list)
    for (idx, sentiment), tup in tweets.items():

        cnt = Counter()
        for token, lang in tup:
            cnt[lang] += 1

        # get the maximum amount of tokens in a dictionary, if multiple maxima exist, this also works
        maximum_keys = [key for m in [max(cnt.values())] for key,val in cnt.items() if val == m]

        if len(maximum_keys) == 1:
            # there is only one maximum length, so append it.
            distribution_lists[sentiment].append(maximum_keys[0])
        else:
            if 'lang1' in maximum_keys and 'lang2' in maximum_keys:
                # equal amount of language1 and language2 tokens in tweet
                distribution_lists[sentiment].append('inconclusive')
            else:
                for idx, lang in enumerate(maximum_keys):
                    if lang == 'lang1' or 'lang2':
                        # append the language instead of other tokens
                        distribution_lists[sentiment].append(maximum_keys[idx])
                    else:
                        # inconclusive as hell
                        distribution_lists[sentiment].append('inconclusive')


    # dictionary with sentiment a keys and COUNTS of languages as values
    distribution_counts = {}
    for key, languages in distribution_lists.items():
        distr = Counter()
        for val in languages:
            distr[val] += 1
        distribution_counts[key] = distr


    return distribution_counts, distribution_lists


def main():

    file_path = '../Sentimix/spanglish_trial.txt'
    tweets = open_file(file_path)

    counts, lists = extract_distribution(tweets)

    for k, value in counts.items():
        print(k)
        for v in value.items():
            print(v)


if __name__ == '__main__':
    main()