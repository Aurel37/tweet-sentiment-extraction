import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
import csv
from utils.text_prep import vectorize, clean
from utils.data_loader import open_csv, get_x_by_label, get_x_not_by_label
from utils.manipulation import *

train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')

#vectorize the text


selected_text = get_x_not_by_label('train.csv', "text", "neutral")
test, d = vectorize(selected_text)
print(len(d))

pauvres = peu_repeter(test, d, 2)
print(len(pauvres))

riches = difference(d, pauvres)


#text_test = np.array(open_csv('test.csv', 'text', 'sentiment'))
#vect_test, d = vectorize(text_test[0], d)
#print("vectorialisation finie")
#res = build(vect_test, d)
#print(res[36])

#begin = slices(text_test[0])
#print(begin[36])
#print("slice")
#final = arange_string(res, begin)
#content = ecrire_resultat(text_test[0], final, text_test[-1])
