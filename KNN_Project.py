#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:17:21 2020

@author: Jean
"""

import numpy as np
import scipy.io as sio
import csv
from utils.data_loader import open_csv
from utils.text_prep import vectorize, clean


train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')
tweets = train[0]
labels = train[2]
data = np.array((tweets, labels))

tweets_0 = [] # neutral tweets
tweets_1 = [] # positive tweets
tweets_2 = [] # negative tweets

for i in range(len(tweets)):
    if labels[i]==0:
        tweets_0.append(tweets[i])
    elif labels[i]==1:
        tweets_1.append(tweets[i])
    else:
        tweets_2.append(tweets[i])
        
neutral_tweets = np.array(tweets_0)
positive_tweets = np.array(tweets_1)
negative_tweets = np.array(tweets_2)

tweets_classes = np.array((neutral_tweets, positive_tweets, negative_tweets))
n = len(tweets_classes[0]) + len(tweets_classes[1]) + len(tweets_classes[2])

def jaccard(str1, str2): 
    if str1 == str2:
        return float(n)
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def jaccard_lettres(str1, str2):
    l1 = []
    l2 = []
    for i in range(len(str1)):
        l1.append(str1[i])
    for i in range(len(str2)):
        l2.append(str2[i])
    a = set(l1)
    # print(a)
    b = set(l2)
    # print(b)
    c = a.intersection(b)
    # print(c)
    return float(len(c)) / (len(a) + len(b) - len(c))

def KNN_word(chaine):
    distances = np.array([0., 0., 0.])
    for phrase in neutral_tweets:
        mots = phrase.split()
        for word in mots:
            distances[0] += jaccard_lettres(chaine, word)
    distances[0] = distances[0] / neutral_tweets.shape[0]
    for phrase in positive_tweets:
        mots = phrase.split()
        for word in mots:
            distances[1] += jaccard_lettres(chaine, word)
    distances[1] = distances[1] / positive_tweets.shape[0]
    for phrase in negative_tweets:
        mots = phrase.split()
        for word in mots:
            distances[2] += jaccard_lettres(chaine, word)
    distances[2] = distances[2] / negative_tweets.shape[0]
    return np.argmax(distances)

def KNN_sentence(chaine):
    chaine = str(chaine)
    similarites = np.array([0., 0., 0.])
    for tweet in neutral_tweets:
        similarites[0] += jaccard(str(tweet), chaine)
    similarites[0] = similarites[0] / neutral_tweets.shape[0]
    for tweet in positive_tweets:
        similarites[1] += jaccard(str(tweet), chaine)
    similarites[1] = similarites[1] / positive_tweets.shape[0]
    for tweet in negative_tweets:
        similarites[2] += jaccard(str(tweet), chaine)
    similarites[2] = similarites[2] / negative_tweets.shape[0]
    return np.argmax(similarites)

def KNN(chaine):
    # chaine = clean(chaine)
    mots = chaine.split()
    classe = 0
    if len(mots) <= 1:
        classe = KNN_word(chaine)
    else:
        classe = KNN_sentence(chaine)
    return(classe)

"""
Premier test pour identifier la partie du tweet qui génère l'émotion

def identifie_partie_emotion(chaine):
    classe = KNN(chaine)
    if classe == 0:
        return("Tweet neutre, pas d'emotions detectee.")
    mots = chaine.split()
    chaine1 = ""
    chaine2 = ""
    for i in range(len(mots)//2):
        chaine1 += mots[i]
    for i in range(len(mots)//2, len(mots)):
        chaine2 += mots[i]+" "
    print(chaine1)
    print(chaine2)
    classe1 = KNN(chaine1)
    classe2 = KNN(chaine2)
    if classe1 != 0 and classe2 != 0:
        return chaine
    if classe1 != 0:
        return chaine1
    else:
        return chaine2
"""
  
def recursive_selection(chaine, original_class):
    classe = KNN(chaine)
    if classe != original_class:
        return ''
    mots = chaine.split()
    if len(mots) == 1:
        return chaine
    chaine1 = ""
    chaine2 = ""
    for i in range(len(mots)//2):
        chaine1 += mots[i]+" "
    for i in range(len(mots)//2, len(mots)):
        chaine2 += mots[i]+" "
    chaine1 = chaine1[:-1]
    chaine2 = chaine2[:-1]
    a = recursive_selection(chaine1, original_class)
    b = recursive_selection(chaine2, original_class)
    if len(a) >= len(b):
        return a
    else:
        return b
    
def simple_selection_bis(chaine, original_class):
    mots = chaine.split()
    if len(mots) == 1:
        return chaine
    chaine1 = ""
    chaine2 = ""
    for i in range(len(mots)//2):
        chaine1 += mots[i]+" "
    for i in range(len(mots)//2, len(mots)):
        chaine2 += mots[i]+" "
    chaine1 = chaine1[:-1]
    chaine2 = chaine2[:-1]
    classe1 = KNN(chaine1)
    classe2 = KNN(chaine2)
    if classe1 != original_class and classe2 != original_class:
        return chaine
    if classe1 == original_class:
        return chaine1
    else:
        return chaine2
    
def simple_selection(chaine, original_class, nb_parties):
    mots = chaine.split()
    if len(mots) == 1:
        return chaine
    resultat = ""
    parties = ["" for k in range(nb_parties)]
    for k in range(nb_parties):
        for i in range(k*len(mots)//nb_parties, (k+1)*len(mots)//nb_parties):
            parties[k] += mots[i]+" "
    for k in range(nb_parties):
        parties[k] = parties[k][:-1]
        classe = KNN(parties[k])
        if classe == original_class:
            resultat += parties[k]
    return resultat

