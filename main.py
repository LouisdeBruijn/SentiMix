import os 
import tools.baseline as bt
import tools.data as data_tools
import tools.baseline as baseline_tools

from sklearn.svm import LinearSVC

dt_path = "./data_files/train_conll_hinglish.txt"

data = data_tools.Data(dt_path)
baseline_tools.test(data)