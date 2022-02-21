import numpy as np
import pandas as pd

## save a smaller csv of the training data
################################################################
df = pd.read_csv('../Data/train.csv')

df_short = df.iloc[0:5000, :]

df_short.to_csv('../Data/train_short.csv', index=False)
################################################################



## Data import; first for coding train_short.csv; final version with train.csv
# dataframe must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order
# df = pd.read_csv("./Data/train.csv").iloc[:, [11, 2, 14]]
# print(df.head())
# df_r = pd.pivot_table(df, index=['user_id'], columns=['media_id'], values=['is_listened'], fill_value=0).astype(int)
