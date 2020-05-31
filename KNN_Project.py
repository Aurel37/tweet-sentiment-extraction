#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 21:17:21 2020

@author: Jean
"""

import numpy as np
import scipy.io as sio
import csv
from utils.text_prep import vectorize, clean


train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')
tweets = train[0]
labels = train[2]
data = np.array((train, labels))

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

def jaccard(str1, str2): 
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

def separe_mots(phrase):
    liste = phrase.split()
    return liste

def KNN_word(chaine):
    distances = np.array([0., 0., 0.])
    for phrase in neutral_tweets:
        mots = separe_mots(phrase)
        for word in mots:
            distances[0] += jaccard_lettres(chaine, word)
    distances[0] = distances[0] / neutral_tweets.shape[0]
    for phrase in positive_tweets:
        mots = separe_mots(phrase)
        for word in mots:
            distances[1] += jaccard_lettres(chaine, word)
    distances[1] = distances[1] / positive_tweets.shape[0]
    for phrase in negative_tweets:
        mots = separe_mots(phrase)
        for word in mots:
            distances[2] += jaccard_lettres(chaine, word)
    distances[2] = distances[2] / negative_tweets.shape[0]
    return np.argmax(distances)

def KNN_sentence(chaine):
    similarites = np.array([0., 0., 0.])
    for tweet in neutral_tweets:
        similarites[0] += jaccard(tweet, chaine)
    similarites[0] = similarites[0] / neutral_tweets.shape[0]
    for tweet in positive_tweets:
        similarites[1] += jaccard(tweet, chaine)
    similarites[1] = similarites[1] / positive_tweets.shape[0]
    for tweet in negative_tweets:
        similarites[2] += jaccard(tweet, chaine)
    similarites[2] = similarites[2] / negative_tweets.shape[0]
    return np.argmax(similarites)

def KNN(chaine):
    mots = separe_mots(chaine)
    classe = 0
    if len(mots) == 0:
        return("Erreur, chaine vide")
    if len(mots) == 1:
        classe = KNN_word(chaine)
    else:
        classe = KNN_sentence(chaine)
    return(classe)
    
def identifie_partie_emotion(chaine):
    classe = KNN(chaine)
    if classe == 0:
        return("Tweet neutre, pas d'emotions detectee.")
    mots = separe_mots(phrase)
    chaine1 = ""
    chaine2 = ""
    for i in range(len(mots)//2):
        chaine1 += mots[i]
    for i in range(len(mots)//2, len(mots)):
        chaine2 += mots[i]
    classe1 = KNN(chaine1)
    classe2 = KNN(chaine2)
    if classe1 != 0 and classe2 != 0:
        return chaine
    if classe1 != 0:
        return chaine1
    else:
        return chaine2
            
        
    
