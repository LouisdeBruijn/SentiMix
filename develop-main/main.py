from bi_lstm import run_model
from baseline import *
from data.data_manager import *


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-tr', '--train_data', help='path to training data')
    parser.add_argument('-etr', '--extra_train_data',
                        help="path to additional data (Spanglish 2016 tweets)")
    parser.add_argument('-te', '--test_data', help='path to testing data')

    parser.add_argument('-b', "--baseline",
                        help='show baseline result', action="store_true")
    parser.add_argument(
        '-s', "--svm", help='show svm result', action="store_true")

    parser.add_argument('-en_emb', '--English_embeddings',
                        help="word embeddings for English")

    parser.add_argument('-es_emb', '--Spanish_embeddings',
                        help="word embeddings for Spanish")

    args = parser.parse_args()

    if args.train_data == None or args.test_data == None:
        print("Data path to training or testing required. See -h for more info.")
        exit()

    train = Data(args.train_data, format="conll")
    test = Data(args.test_data, format="conll")

    if args.extra_train_data != None:
        addt = Data(args.extra_train_data, format="json")
        train = Preprocessor.combine_data(train, addt)

    if args.baseline:
        print("Running random guessing baseline...")
        run_random_baseline(train, test)

    if args.svm:
        print("Running TFIDF SVM...")
        run_baseline_tfidf(train, test)

    # Neural model
    print("Runinning Neural model...")
    run_model(train, test, args.English_embeddings, args.Spanish_embeddings)
