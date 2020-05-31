import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


def clean(tweet):
    """return a clean string
    tweet : string"""
    if type(tweet) == str:
        clean_p = [word for word in tweet if word not in string.punctuation]
        clean_p = ''.join(clean_p)
        clean_p = clean_p.split(' ')
        return ' '.join([word for word in clean_p if word.lower() not in stopwords.words('english') +['im', 'day','get','go', 'dont', 'today', 'going', 'got', 'work', 'one', 'cant', 'time', 'know', 'back', 'really', 'see', 'mothers', 'want', 'home', 'night', 'still', 'new', 'think', 'much', 'well', 'thanks', 'last', 'morning', 'need', 'tomorrow'] ])
    else:
        return ' '


def vectorize(text_tab, feature_names=None):
    """return a numpy digit number array
    text_tab : numpy array of strings
    """
    process = []
    for tweet in text_tab:
        if type(tweet) == str:
            process.append(clean(tweet))
    process = np.array(process)
    vect = CountVectorizer()
    if feature_names is None:
        vect.fit(process)
    else:
        vect.vocabulary = feature_names
    res = vect.transform(process)
    return res.toarray(), vect.get_feature_names()
