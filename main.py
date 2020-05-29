import numpy as np
import matplotlib.pyplot as plt
from utils.metric import jaccard
from utils.text_prep import vectorize
from utils.data_loader import open_csv

train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')

selected = train[1]
#vectorize the text

# /!\ the sets text and selected_text are not vectorized on the same word base yet
#text_train, translation = vectorize(train[0])

def histo_repartition(text, words, N):
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
    plt.xticks(x, popular_words)
    plt.plot(x, values)
    plt.title("Positive")
    plt.show()
                     
    negative_index = np.argsort(negative)
    negative_index = np.flipud(negative_index) 
    values = [negative[negative_index[i]] for i in range(N)]
    popular_words = [words[negative_index[i]] for i in range(N)]
    plt.xticks(x, popular_words )
    plt.plot(x, values)
    plt.title("Negative")
    plt.show()

                     
    neutral_index = np.argsort(neutral)
    neutral_index = np.flipud(neutral_index) 
    values = [neutral[neutral_index[i]] for i in range(N)]
    popular_words = [words[neutral_index[i]] for i in range(N)]
    plt.xticks(x, popular_words )
    plt.plot(x, values)
    plt.title("Neutral")
    plt.show()


def histo_size(texts, labels):
    
    sizes_positive =  []
    sizes_negative =  []
    sizes_neutral =  []
    for i in range(len(texts)):
        if labels[i] == 0:
            #print(i)
            sizes_neutral.append(len(texts[i]))
        elif labels[i] == 1:
            
            sizes_positive.append(len(texts[i]))
        else:
            sizes_negative.append(len(texts[i]))
    n, bins, patches = plt.hist(x=sizes_neutral, bins=max(sizes_neutral)+1, color='#0504aa',
                                alpha=0.7, rwidth=0.8)
    plt.title("Neutral size")
    plt.show()
    n, bins, patches = plt.hist(x=sizes_positive, bins=max(sizes_positive)+1, color='#0504aa',
                                alpha=0.7, rwidth=0.8)
    plt.title("Positive size")
    plt.show()
    n, bins, patches = plt.hist(x=sizes_negative, bins=max(sizes_negative)+1, color='#0504aa',
                                alpha=0.7, rwidth=0.8)
    plt.title("Negative size")
    plt.show()

print(train[1][314])
print(len(train[1]))
#histo_size(train[1], train[-1])
#histo_repartition(text_train, translation, 20)

#text_train, feature_names  = vectorize(train[0][:10])
#selected_text_train, d = vectorize(train[1][:10], feature_names)
