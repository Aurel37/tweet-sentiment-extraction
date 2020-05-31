import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
from scipy.sparse import csr_matrix
import csv
from utils.text_prep import vectorize, clean
from utils.data_loader import get_x_not_by_label, open_csv
from utils.manipulation import *
from utils.classification import log_reg
from utils.reduction import PCA, standardize
from utils.SVM import *
from KNN_Project import simple_selection, simple_selection_bis, recursive_selection


def vectorize_pca(document, dimpca, x_col, lb_col):
    text = get_x_not_by_label(document, x_col, 'neutral')
    selected_text = get_x_not_by_label(document, lb_col, 'neutral')
    test, d = vectorize(text)
    pauvres, histo = peu_repeter(test, d, 5)
    riches = difference(d, pauvres)
    text_array_clean, dico = vectorize(text, riches)
    sel_text_array, dico = vectorize(selected_text, riches)
    text_array_st = standardize(text_array_clean)
    text_array_pca = PCA(text_array_st, dimpca)
    return text_array_pca, sel_text_array


def classification(document, dimpca, x_col, lb_col):
    X, Y = vectorize_pca(document, dimpca, x_col, lb_col)
    n = len(X)
    classify = log_reg()
    r = np.random.permutation(n)
    X_train = X[r[:n]]
    Y_train = Y[r[:n]]
    m = len(Y_train[0])
    res = np.zeros((m, dimpca+1))
    for i in range(m):
        classify.fit(X_train, Y_train[:, i])
        res[i] = classify.beta
    np.savetxt('test.txt', res)
    return res




def sparse():
    text_train = get_x_not_by_label('train.csv', "text", "neutral")
    text, d = vectorize(text_train)
    sparse_text = csr_matrix(text)
    print(sparse_text.getnnz())


def treatment(positives, negatives):
    positives_treated = []
    negatives_treated = []
    for i in range(len(positives)):
        positives_treated.append(simple_selection(positives[i], 1))
    for i in range(len(negatives)):
        negatives_treated[i].append(simple_selection(negatives[i], 2))
    final_list = []
    final_list.append(positives_treated)
    final_list.append(negatives_treated)
    return final_array

def algoo_KNN():
    """
    Méthode KNN
    """
    test = open_csv('test.csv', 'text', 'sentiment')
    tweets = test[0]
    labels = test[1]
    data = np.array((tweets, labels))

    tweets_neutral = [] # neutral tweets
    tweets_positive = [] # positive tweets
    tweets_negative = [] # negative tweets

    for i in range(len(tweets)):
        if labels[i]==0:
            tweets_neutral.append(tweets[i])
        elif labels[i]==1:
            tweets_positive.append(tweets[i])
        else:
            tweets_negative.append(tweets[i])

    tweets_classes = np.array((tweets_neutral, tweets_positive, tweets_negative))
    n = len(tweets_classes[0]) + len(tweets_classes[1]) + len(tweets_classes[2])

    # Treament of tweets : 
    treated_list = treatment(positive_tweets, negative_tweets)




def algo_naif():
    """
    Méthode naïve
    """
    selected_text = get_x_not_by_label('train.csv', "text", "neutral")
    test, d = vectorize(selected_text)
    pauvres, histo = peu_repeter(test, d, 5)
    riches = difference(d, pauvres)
    text_test = np.array(open_csv('test.csv', 'text', 'sentiment'))
    vect_test, d = vectorize(text_test[0], riches)
    res = build(vect_test, d)
    begin = slices(text_test[0])
    final = arange_string(res, begin)
    content = ecrire_resultat(text_test[0], final, text_test[-1])



