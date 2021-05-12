import csv
import sys
import string
import re
import os

import nltk 
from nltk.stem import PorterStemmer

def myMap(n):
  return str(n)

def string_process(words,stopwords,stem = False):
  ps = PorterStemmer()
  words = words.replace("<br />", " ").rstrip()
  words = words.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
  words = words.translate(str.maketrans('', '', string.punctuation))
  words = words.lower()
  words = words.split()
  if stem:
    words = [ps.stem(word) for word in words]
  words = [word for word in words if word not in stopwords]
  return words

def strip_punc(s):
  return s.translate(str.maketrans('', '', string.punctuation))

def set_stopwords():
  stopwords = open("/content/drive/My Drive/translations/stopwords.en.txt", "r", encoding="utf8")
  words = stopwords.read().split("\n")
  stopwords.close()
  return words

def top_n_score(X,y,model,n,a2i,i2a):

  scores = model.predict_proba(X)
  found = 0
  for i,probs in enumerate(scores):
    s = list( zip( probs, a2i.keys() ) )
    s.sort(reverse=True)
    top = s[0:n]
    guesses = [i[1] for i in top]
    
    ans = i2a[y[i]]

    if ans in guesses:
      found += 1
  return found/X.shape[0]