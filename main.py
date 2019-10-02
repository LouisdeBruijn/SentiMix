import os 
import tools.baseline as bt
import tools.data as data_tools
import tools.baseline as baseline_tools
import numpy as np

from sklearn.svm import LinearSVC

dt_path = "./data_files/train_conll_hinglish.txt"

# data = data_tools.Data(dt_path, shuffle=True)
# baseline_tools.test(data)
# baseline_tools.plot_svm_accuracy(data)

layer_size = [2,3,5,2]
bias = [np.zeros((s,1)) for s in layer_size[1:]]
for b in bias:
	print(b)