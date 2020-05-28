import numpy as np
from utils.metric import jaccard
from utils.text_prep import vectorize
from utils.data_loader import open_csv

train = open_csv('train.csv', 'text', 'selected_text', 'sentiment')

#vectorize the text
# /!\ the sets text and selected_text are not vectorized on the same word base yet
text_train = vectorize(train[0])
