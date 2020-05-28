import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string


def clean(tweet):
    """return a clean string
    tweet : string"""
    clean_p = [word for word in tweet if word not in string.punctuation]
    clean_p = ''.join(clean_p)
    clean_p = clean_p.split(' ')
    return ' '.join([word for word in clean_p if word.lower() not in stopwords.words('english')])


def vectorize(text_tab):
    """return a numpy digit number array
    text_tab : numpy array of strings
    """
    process = []
    for tweet in text_tab:
        if type(tweet) == str:
            process.append(clean(tweet))
    process = np.array(process)
    vect = CountVectorizer()
    vect.fit(process)
    res = vect.transform(process)
    return res.toarray()
