import os 
import tools.baseline as bt
import tools.data as data_tools
import tools.baseline as baseline_tools
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

dt_path = "./data_files/spanglish_trial.txt"
dt_embeddings = "./data_files/embeddings.json"

data = data_tools.Data(dt_embeddings, dt_path, shuffle=True, split=0.7)
baseline_tools.test_one(KNeighborsClassifier(),data)
# baseline_tools.plot_svm_accuracy(data)
# vec = baseline_tools.word_embedding_vectorizor(['hello','world'], dt_embeddings)

