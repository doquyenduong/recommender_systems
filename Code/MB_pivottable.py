from deezerData import readData
import pandas as pd

df_import, _, _, _, _, _, _ = readData()
print(df_import.info())
df = df_import.iloc[:, [11, 2, 14]]
# df = pd.read_csv('../Data/train_short.csv').iloc[:, [11, 2, 14]]
print(df.info(), df.shape)
df_r = pd.pivot_table(df, index=['user_id'], columns=['media_id'], values=['is_listened'], fill_value=-1).astype(int)
print(df_r.head())
df_r.to_csv('./pivot.csv')
matrix_r = df_r.to_numpy(dtype='int8')
print(matrix_r)


R = matrix_r