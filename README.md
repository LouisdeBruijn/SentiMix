# SentiMix Shared Task

Providing a space for development of our scripts for the SentiMix 2020 shared task.

Find it here https://competitions.codalab.org/competitions/20654#learn_the_details-overview


## Getting started

First download the english and spanish fastText word embeddings:

They can be found at https://fasttext.cc/docs/en/aligned-vectors.html

Then install the necessary libraries:

```
pip install -r requirements.txt
```

To run the final system:

The traininig data should be in the format of conll. Except the conv_2016_spanglish_annotated.json.

```
cd develop-main
python3 main.py -tr [PATH-TO-TRAINNING-DATA] -te [PATH-TO-TESTING-DATA] -en_emb [PATH-TO-ENGLISH-WORD-EMBEDDINS] -eS_emb [PATH-TO-SPANISH-WORD-EMBEDDINS] -b -s
```

For more information on the arguments:

```
python3 main.py -h
```
