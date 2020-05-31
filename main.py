import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
import csv
from utils.text_prep import vectorize, clean
from utils.data_loader import open_csv, get_x_by_label, get_x_not_by_label
from utils.manipulation import *

train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')

#vectorize the text


def vectorize_pca(document, column, lb, dimpca):
    text = get_x_not_by_label('train.csv', 'test', 'neutral')
    test, d = vectorize(text)
    pauvres, histo = peu_repeter(test, d, 5)
    riches = difference(d, pauvres)
    text_array_clean, dico = vectorize(text, riches)
    text_array_st = standardize(text_array_clean)
    text_array_pca = PCA(text_array_st, dimpca)
    return text_array_pca

#N = 20
#test, d = vectorize(train[0])
#histo_repartition(test, d, N, train[-1])

selected_text = get_x_not_by_label('train.csv', "text", "neutral")
test, d = vectorize(selected_text)
print(len(d))

pauvres, histo = peu_repeter(test, d, 5)
print(histo)
make_histo(histo, "Histogramme des repetitions des mots de notre ensemble", "blue",
           "Nombre de répétitions des mots", "nombre de mots concernés")
print(len(pauvres))

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
