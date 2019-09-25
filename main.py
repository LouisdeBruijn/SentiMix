import os 
import baseline_tools.baseline as bs
from sklearn.svm import LinearSVC

dt_path = "train_conll_hinglish.txt"

bs.test(dt_path)