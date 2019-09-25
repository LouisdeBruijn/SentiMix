class Data:
	def __init__(self, train_set, test_set = None, split=0.5, combine = False):
		self.x_train, self.y_trian, self.x_test, self.y_test = self.get_data(train_set, test_set, split, combine)

	#read the data from codalab
	def read_data(self, path):
		docs = []
		labels = []
		file = open(path, 'r')
		start = True
		sentence = []
		for line in file:
			if line != "\n":
				if start:
					sentence = []
					start = False
					l = line.split()
					#this adds the labels
					labels.append(l[len(l)-1])
				else:
					w = line.strip().split()

					if len(w) < 2:
						continue

					#only assigning the actual word for now.
					sentence.append(w[0])
			else:
				start = True
				docs.append(sentence)
				continue
		return docs, labels

	def get_data(self, dataset_path, testdata_path = None, split = 0.5, combine=False):
		'''Automatically output Xtrain Ytrain Xtest and Ytest'''
		X,Y = self.read_data(dataset_path)
		split_point = int(split * len(X))
		
		if testdata_path == None:
			Xtrain = X[:split_point]
			Ytrain = Y[:split_point]

			Xtest = X[split_point:]
			Ytest = Y[split_point:]
		else:
			Xtrain = X
			Ytrain = Y
			Xtest, Ytest = self.read_data(testdata_path)
			if combine:
				Xtrain = X + Xtest
				Ytrain = Y + Ytest

		return Xtrain, Ytrain, Xtest, Ytest