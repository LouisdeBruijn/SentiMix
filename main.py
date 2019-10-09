import os 
import tools.baseline as bt
import tools.data as data_tools
import tools.baseline as baseline_tools
import numpy as np

from sklearn.svm import LinearSVC

dt_path = "./data_files/train_conll_spanglish.txt"

data = data_tools.Data(dt_path, shuffle=True, split=0.8)
baseline_tools.test(data)
baseline_tools.plot_svm_accuracy(data)


