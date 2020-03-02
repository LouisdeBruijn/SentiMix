Data Format

Hinglish trial data: 1,870 samples
Spanglish trial data: 2,000 samples

We follow the CoNLL format for both datasets. Every token in a tweet has its own line, and next to the token you will find its corresponding language identification label separated by a tab. The first line of a tweet contains tab-separated metadata where we provide the index and the sentiment of the tweet. Every tweet is separated by an empty line. Here is an example of two tweets from the Spanglish dataset, one positive and one neutral:

meta	1	positive
So	lang1
that	lang1
means	lang1
tomorrow	lang1
cruda	lang2
segura	lang2
lol	lang1

meta	2	neutral
Tonight	lang1
peda	lang2
segura	lang2



Official Competition Metric for the Task

The metric for evaluating the participating systems will be as follows. We will use F1 averaged across the positives, negatives, and the neutral. The final ranking would be based on the average F1 score. However, for further theoritical discussion and we will release macro-averaged recall (recall averaged across the three classes), since the latter has better theoretical properties than the former2015), and since this provides better consistency. Each participating team will initially have access to the training data only. Later, the unlabelled test data will be released. After SemEval-2020, the labels for the test data will be released as well. We will ask the participants to submit their predictions in a specified format (within 24 hours), and the organizers will calculate the results for each participant. We will make no distinction between constrained and unconstrained systems, but the participants will be asked to report what additional resources they have used for each submitted run.
