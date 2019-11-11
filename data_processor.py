import gc
import sys
import os
import re
import gensim
from nltk import RegexpTokenizer, TweetTokenizer
from emoji import UNICODE_EMOJI


class Data:
	
	path = ""
	labels = []
	documents = []

	def __init__(self, path):
		self.path = path
		self.__load()

	def __load(self):
		print("loading data...", end="")
		with open(self.path, "r") as file:
			docs = []
			sentence = []
			for row in file:
				if row == "\n":
					docs.append(sentence)
					sentence = []
				else:
					s = str(row).split('\t')
					if len(s) >= 3:
						continue
					sentence.append(s[0])
		
		self.documents = docs
		print_done()

	def clean(self):
		print("cleaning texts...", end="")
		docs = []
		for doc in self.documents:
			doc = " ".join(doc)

			doc = re.sub(r'[.!?\\//…@"“]', '' , doc)
			doc = re.sub(r"[‘']", '', doc)

			tokenizer = TweetTokenizer(reduce_len=True)
			doc = tokenizer.tokenize(doc)

			docs.append(doc)
			

		self.documents = docs
		print_done()

	def mask_emoji(self):
		# Mask the emoji to positive, negative and neutral
		docs = self.documents
		for doc in docs:
			for i, token in enumerate(doc):
				if token in UNICODE_EMOJI:
					print(token)


	def __isStringEmpty(self, item):
		return item != ''


def print_done():
	sys.stdout.write("....done!\n")
	sys.stdout.flush()

def create_emoji_data(data):
	path = "emoji_sentiment.txt"
	emojis = []
	with open(path, "w") as file:
		for doc in data.documents:
			for token in doc:
				if token in UNICODE_EMOJI:
					dic = [w[0] for w in emojis]
					if token in dic:
						continue
					
					print(token)
					i = input("""Enter 'p' for pos, 'n" for 'neg' and any for 'neutral'""")
					if i == 'p':
						emojis.append([token, "positive"])
					elif i == 'n':
						emojis.append([token, "negative"])
					else:
						emojis.append([token, "neutral"])
		
		file.writelines([" ".join(w) + "\n" for w in emojis])

def load_emoji_data():
	path = "emoji_setiment.txt"
	with open(path, "r") as file:
		for row in file:
			print(row)
		pass

def explore_embeddings(data, embeddings):

	unknown_words = []
	for doc in data.documents:
		for token in doc:
			try:
				word = embeddings[token]
			except KeyError:
				unknown_words.append(token)	

	print(unknown_words)
	print(f"There are a total of {len(unknown_words)} unkown words in the embeddings.")

if __name__ == "__main__":
	gc.enable()
	path = "data_files/train_conll_hinglish.txt"
	os.system("clear")
	
	# Loading the data 
	data = Data(path)

	# Retokenize the data with nltk tweet tokenizer
	data.clean()

	# Load the embeddings
	embedding_path = "data_files/wiki.en.align.vec"

	print("Loading embeddings...")
	embeddings = gensim.models.KeyedVectors.load_word2vec_format(embedding_path)

	explore_embeddings(data, embeddings)

	# create_emoji_data(data)

	# print(data.documents[:100])
	# Masking emojis



