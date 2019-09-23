import os 
import baseline_tools.baseline as bs

dt_path = "train_conll_spanglish.txt"

bs.test(dt_path,use_cv=True)