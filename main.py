import numpy as np
from utils.metric import jaccard
from utils.text_prep import vectorize
from utils.data_loader import open_csv

train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')

#vectorize the text

text_train, feature_names  = vectorize(train[0][:10])
selected_text_train, d = vectorize(train[1][:10], feature_names)
