import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
from scipy.sparse import csr_matrix
import csv
from utils.text_prep import vectorize, clean
from utils.data_loader import open_csv, get_x_by_label, get_x_not_by_label
from utils.manipulation import *
from utils.SVM import *
from KNN_Project import simple_selection, simple_selection_bis, recursive_selection


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

"""
Tentative de SVM
N = 20
test, d = vectorize(train[0])
histo_repartition(test, d, N, train[-1])

text_train = get_x_not_by_label('train.csv', "text", "neutral")
text, d = vectorize(text_train)
pauvres, histo = peu_repeter(text, d, 5)
riches = difference(d, pauvres)
Xtrain, riches = vectorize(text_train, riches) 
print("vec")
selected_text = get_x_not_by_label('train.csv', "selected_text", "neutral")
Ytrain, riches = vectorize(selected_text, riches)
eta = 0.0001
lambada = 0.001n = 1
SAG = SAGRegression(lambada, eta)
print("z'est partiii")
w, b, L = SAG.fit(Xtrain, Ytrain,epochs=n)

"""

text_train = get_x_not_by_label('train.csv', "text", "neutral")
text, d = vectorize(text_train)
print(len(text)*len(d))
sparse_text = csr_matrix(text)
nb_coef = 0
for i in range(sparse_text.shape[0]):
    nb_coef += sparse_text[i].shape[0]

print(sparse_text[0])
print(sparse_text[0].shape)


"""

# treats the lists of positive and negatives tweets 
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
# treated_list = treatment(positive_tweets, negative_tweets)


"""


"""
Algo naîf de construction de solution:

selected_text = get_x_not_by_label('train.csv', "text", "neutral")
test, d = vectorize(selected_text)
print(len(d))

pauvres, histo = peu_repeter(test, d, 5)
print(histo[0:100])
make_histo(histo, "Histogramme des repetitions des mots de notre ensemble", "blue")
print(len(pauvres))

riches = difference(d, pauvres)


text_test = np.array(open_csv('test.csv', 'text', 'sentiment'))
vect_test, d = vectorize(text_test[0], riches)
print("vectorialisation finie")
res = build(vect_test, d)
print(res[36])

begin = slices(text_test[0])
print(begin[36])
print("slice")
final = arange_string(res, begin)
content = ecrire_resultat(text_test[0], final, text_test[-1])
"""


