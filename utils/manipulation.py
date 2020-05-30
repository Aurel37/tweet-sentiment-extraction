import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
import csv
from utils.text_prep import vectorize, clean


def make_histo(histo, title, col):
    n, bins, patches = plt.hist(x=histo, bins=40, color=col,
                                alpha=0.7, rwidth=0.8, range = (0, 100))
    plt.title(title)
    plt.show()

def histo_repartition(text, words, N):
    """
    Calcul l'histogramme des N plus influents mots en fonction de leur label.
    """
    positive = np.zeros(text.shape[1])
    negative = np.zeros(text.shape[1])
    neutral = np.zeros(text.shape[1])
    for l in range(len(text)):
        if train[-1][l] == 0:
            neutral += text[l]
        elif train[-1][l] == 1:
            positive += text[l]
        else:
            negative += text[l]
    positive_index = np.argsort(positive)
    positive_index = np.flipud(positive_index)
    values = [positive[positive_index[i]] for i in range(N)]
    popular_words = [words[positive_index[i]] for i in range(N)]
    x = np.arange(N)
    #plt.xticks(x, popular_words)
    plt.plot(x, values)
    plt.title("Positive")
    plt.show()
    print(popular_words)

    negative_index = np.argsort(negative)
    negative_index = np.flipud(negative_index)
    values = [negative[negative_index[i]] for i in range(N)]
    popular_words = [words[negative_index[i]] for i in range(N)]
    #plt.xticks(x, popular_words)
    plt.plot(x, values)
    plt.title("Negative")
    plt.show()
    print(popular_words)

    neutral_index = np.argsort(neutral)
    neutral_index = np.flipud(neutral_index)
    values = [neutral[neutral_index[i]] for i in range(N)]
    popular_words = [words[neutral_index[i]] for i in range(N)]
    #plt.xticks(x, popular_words)
    plt.plot(x, values)
    plt.title("Neutral")
    plt.show()
    print(popular_words)






def histo_size(texts, labels, col):
    """
    Calcul l'histogramme des tailles des tweets en fonction de leur label
    """
    
    sizes_positive =  []
    sizes_negative =  []
    sizes_neutral =  []
    for i in range(len(texts)):
        if type(texts[i]) == float:
            texts[i] = ''
        if labels[i] == 0:
            #print(i)
            sizes_neutral.append(len(texts[i]))
        elif labels[i] == 1:
            
            sizes_positive.append(len(texts[i]))
        else:
            sizes_negative.append(len(texts[i]))
    print(sizes_neutral[: 20])
    n, bins, patches = plt.hist(x=sizes_neutral, bins=max(sizes_neutral)+1, color=col,
                                alpha=0.7, rwidth=0.8)
    plt.title("Neutral size")
    plt.show()
    n, bins, patches = plt.hist(x=sizes_positive, bins=max(sizes_positive)+1, color=col,
                                alpha=0.7, rwidth=0.8)
    plt.title("Positive size")
    plt.show()
    n, bins, patches = plt.hist(x=sizes_negative, bins=max(sizes_negative)+1, color=col,
                                alpha=0.7, rwidth=0.8)
    plt.title("Negative size")
    plt.show()


def build(vect, translate):
    """
    Reconstruit l'ensemble des mots de vect present dans le dico translate
    """
    res = []
    for i in range(len(vect)):
        selected = []
        for j in range(len(vect[0])):
            if vect[i][j] > 0:
                selected.append(translate[j])
        res.append(selected)
    return res

def arange_string(res, begin):
    """
    Reconstruit dans l'ordre les selections des twits 
    """
    final = []
    for i in range(len(res)):
        s = ''
        for j in range(len(begin[i])):
            if begin[i][j].lower() in res[i]:
                if len(s) > 0:
                    s += " " + begin[i][j]
                else:
                    s += begin[i][j]
        final.append(s)
    return final


def slices(strings_brut):
    """
    Separe tweets de strings_brut en liste de mot
    On clean egalement les tweets
    """
    strings = np.copy(strings_brut)
    sliced = []
    for i in range(len(strings)):
        strings[i]= clean(strings[i])
        sliced.append(strings[i].split(" "))
    return sliced

def ecrire_resultat(begin, final, label):
    """
    Ecrit le resultat dans un csv
    """

    content = [[" Text", " Selected_text", "Sentiments"]]
    for i in range(len(begin)):
        if label[i] == 0:
            content.append([begin[i],  begin[i], label[i]])
        elif len(final[i]) > 0:
            content.append([begin[i],  final[i], label[i]])
        else:
            content.append([begin[i],  "ELLE EST BONNE SA MERE", label[i]])
    print(type(content))
    with open('result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(content)
    return content

def traduction(vect, d):
    """
    On retrouve les mots de vect dans le dico d
    """
    s = ""
    for i in range(len(vect)):
        if vect[i] >0:
            s += d[i]
    return s

def peu_repeter( text, dico, M):
    """
    Donne l'ensemble des mots apparaissant au maximum M fois dans l'ensemble
    des tweets de text 
    """
    histo = np.zeros(len(dico))
    pauvre = []
    for l in text:
        histo += l
    for j in range(len(dico)):
        if histo[j] <= M:
            pauvre.append(dico[j])
    return pauvre, histo

def difference(a, b):
    """
    Difference ensembliste entre a et b sous l'hypothese que b inclus dans a 
    """
    c = [] 
    for i in a:
        if not i in b:
            c.append(i)
    return c
