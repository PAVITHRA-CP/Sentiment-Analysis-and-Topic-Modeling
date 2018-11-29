# Header Functions
import pandas as pd
import numpy as np
import nltk.data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
from datetime import datetime
from gensim import corpora
import datetime as dt
import pickle
import gensim
import random
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import csv
import spacy
from spacy.lang.en import English
import os
import re

#..................................................................................................................
# Initialization

content = []	
parser = English()
i=3
d = {}
sid = SentimentIntensityAnalyzer()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
en_stop = set(nltk.corpus.stopwords.words('english'))
text_data = []
n=2
s=1
d1 = 2016
q =[] 
d2 = {}
#.......................................................................................................................
# Function calls

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma



def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

#..................................................................................................................
# Starts from here

data = pd.read_csv("path to dataset")
datas = data.values
aa = np.array(datas)
data = pd.DataFrame(aa)
data[1]= pd.to_datetime(data[1]) # sorting data w.r.t date
data = data.sort_values(1)
data.to_csv('new_file.csv', index=False)
f = open('new_file.csv')
path = os.path.realpath(f.name)
quar= pd.read_csv(path)
quar1 = quar.values
qq = np.array(quar1)
quar = pd.DataFrame(qq)
for p in quar[1]:
	l = p.split('-')
	a = int(l[0])
	b = int(l[1])
	c = int(l[2])
	q.append(pd.Timestamp(dt.date(a,b,c)).quarter) # Finding quarter of each date in date field
quar['5'] = q
quar.to_csv('new_file1.csv', index=False)
#.....................................................................................................................
#Calculate sentence polarity

f1 = open('new_file1.csv')
path1 = os.path.realpath(f1.name)
count = 0
while d1<2019:
	while s<5:
		while i<5:
			with open(path1, 'r') as inp :
				for row in csv.reader(inp) :
					a1 = row[1].split('-')
					a2 = int(a1[0])
				  	if int(row[5]) == int(s) and a2 == int(d1):
						text = row[i]
						text1 = unicode(text, errors='ignore') 
						sentences = tokenizer.tokenize(text1)
						for sentence in sentences:
							scores = sid.polarity_scores(sentence)
							for key, score in scores.items():
				    				if key == 'pos':
									 d.setdefault(key, []).append(score)
								elif key == 'neg' :
									d.setdefault(key, []).append(score)
								elif key == 'neu':
									d.setdefault(key, []).append(score)
								else :
									d.setdefault(key, []).append(score)
			

			result =  {key: round(sum(values),2) for key, values in d.items()}
			if i == 3:
				print 'Pros intensity of quarter', s, 'for year', d1, 'is', result
			else :
				print 'Cons intensity of quarter',s, 'for year', d1, 'is', result
			i = i+1
			d.clear()
		s = s+1
		i = 3
	d1 = d1+1
	s = 1
	i = 3
#............................................................................................................................
#Topic Modeling

while n<5 :
	with open(path, 'r') as f :
		 for row in csv.reader(f):
		   	line = row[n]
			line1 = unicode(line, errors='ignore')
			tokens = prepare_text_for_lda(line1)
			if random.random() > .99:
			    #print(tokens)
			    text_data.append(tokens)
	dictionary = corpora.Dictionary(text_data)
	print (dictionary)
	corpus = [dictionary.doc2bow(text) for text in text_data]
	pickle.dump(corpus, open('corpus.pkl', 'wb'))
	dictionary.save('dictionary.gensim')
	NUM_TOPICS = 5
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
	ldamodel.save('model5.gensim')
	topics = ldamodel.print_topics(num_words=6)
	if n == 2 :
		print('\n')
		print('Related to Title:')
		print('\n')
	elif n==3:
		print('\n')
		print('Related to Pros:')
		print('\n')
	else :
		print('\n')
		print('Related to Cons:')
		print('\n')
	for topic in topics:
	    print (topic)
	    content.append(topic)
	n = n+1
#................................................................................................................................

