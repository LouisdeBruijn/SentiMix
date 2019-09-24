import os 
import baseline_tools.baseline as bs
from sklearn.svm import LinearSVC

dt_path = "train_conll_spanglish.txt"

# bs.test_one(LinearSVC(),dt_path,use_cv=True)
bs.__read_data__(dt_path)