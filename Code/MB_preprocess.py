# https://udemy.com/recommender-systems
# https://deeplearningcourses.com/recommender-systems
from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz

# load in the data
df1 = pd.read_csv('../Data/train_short.csv').iloc[:, [0, 2, 6, 9, 10, 11, 14]]
print(df1.info())
d1 = df1.groupby('user_id')['is_listened'].sum()
d2 = d1[d1 > 15].index
df = df1[df1.user_id.isin(d2)]
print(df.shape)
N = df.user_id.max() + 1 # number of users
M = df.media_id.max() + 1 # number of movies
print(N, M)
# split into train and test
df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

A = lil_matrix((N, M))
print("Calling: update_train")
count = 0
def update_train(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/cutoff))

    i = int(row.user_id)
    j = int(row.media_id)
    A[i,j] = row.is_listened
df_train.apply(update_train, axis=1)

# mask, to tell us which entries exist and which do not
A = A.tocsr()
mask = (A > 0)
save_npz("Atrain.npz", A)

# test ratings dictionary
A_test = lil_matrix((N, M))
print("Calling: update_test")
count = 0
def update_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count)/len(df_test)))

    i = int(row.user_id)
    j = int(row.media_id)
    A_test[i,j] = row.is_listened
df_test.apply(update_test, axis=1)
A_test = A_test.tocsr()
mask_test = (A_test > 0)
save_npz("Atest.npz", A_test)
