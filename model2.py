#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:38:21 2018

@author: chendongcai
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import functools
import numpy as np
import random as rd
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import datasets

#%%
data_train=pd.read_csv('/Files/Projects/Kaggle/Pricing Suggestion Challenge/train.tsv',delimiter='\t')
data_test=pd.read_csv( '/Files/Projects/Kaggle/Pricing Suggestion Challenge/test.tsv',delimiter='\t')
stop = set(stopwords.words('english'))

def tokenize(text):

    try: 
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        text = regex.sub(" ", text) # remove punctuation        
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stop, tokens))
        filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        
        return filtered_tokens
            
    except TypeError as e: print(text,e)
#%%
data_train['tokens']=data_train.item_description.map(tokenize)