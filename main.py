import os 
import baseline_tools.baseline as bt
from sklearn.svm import LinearSVC

dt_path = "train_conll_hinglish.txt"

bt.test(dt_path)