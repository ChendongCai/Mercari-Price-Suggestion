#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:05:52 2018

@author: chendongcai
"""

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?',]
vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()
vectorizer.fit_transform(corpus).toarray()