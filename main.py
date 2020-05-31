import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
import csv
from utils.text_prep import vectorize
from utils.data_loader import get_x_not_by_label
from utils.manipulation import peu_repeter, difference
from utils.classification import log_reg
from utils.reduction import PCA, standardize


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
    res = np.zeros((n, dimpca+1))
    for i in range(n):
        classify.fit(X_train, Y_train[:,i])
        res[i] = classify.beta
    np.savetxt('test.txt', res)
    return res


classification('train.csv', 25, 'text', 'selected_text')
#selected_text = get_x_not_by_label('train.csv', "text", "neutral")
#test, d = vectorize(selected_text)
#print(len(d))

#pauvres, histo = peu_repeter(test, d, 5)
#print(histo[0:100])
#make_histo(histo, "Histogramme des repetitions des mots de notre ensemble", "blue")
#print(len(pauvres))

#riches = difference(d, pauvres)


#text_test = np.array(open_csv('test.csv', 'text', 'sentiment'))
#vect_test, d = vectorize(text_test[0], riches)
#print("vectorialisation finie")
#res = build(vect_test, d)
#print(res[36])

#begin = slices(text_test[0])
#print(begin[36])
#print("slice")
#final = arange_string(res, begin)
#content = ecrire_resultat(text_test[0], final, text_test[-1])
