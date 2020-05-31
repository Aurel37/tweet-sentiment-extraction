import pandas as pd
import numpy as np
from utils.text_prep import clean
from sklearn.feature_extraction.text import HashingVectorizer


data = pd.read_csv('train.csv')

selected_text = data['selected_text']
selected_text = selected_text.to_numpy()

clea_selected = []
for i in selected_text:
    clea_selected.append(clean(i))
clea_selected = np.array(clea_selected)

vect = HashingVectorizer()
vect.fit(clea_selected)
X = vect.transform(clea_selected).toarray()
print(X.shape)
