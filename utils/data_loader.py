"""open the csv document and convert them into numpy array"""
from __future__ import absolute_import
import pandas as pd


def open_csv(path, *col):
    """return the data converted into a numpy array in a set
    path : path of the csv document
    col : columns that may be interesting"""
    data = pd.read_csv(path)
    data['sentiment'] = data['sentiment'].map({'neutral':0, 'positive':1, 'negative':2})
    res = []
    for i in col:
        res.append(data[i].to_numpy())
    return res
