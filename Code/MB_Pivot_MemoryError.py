import pandas as pd

df = pd.read_csv("../Data/train_short.csv").iloc[:, [11, 2, 14]]
print(df)
#df_r = pd.pivot_table(df, index=['user_id'], columns=['media_id'], values=['is_listened'], fill_value=-1).astype(int)

print(len(df))
user_unique = list(set(df.user_id))
media_unique = list(set(df.media_id))
# print(user_unique)
# print(media_unique)

df_pivot_manual = pd.DataFrame(data=None, index=user_unique, columns=media_unique, dtype='float').fillna(-1)

for i in range(len(df)-1):
    df_pivot_manual.loc[df['user_id'][i], df['media_id'][i]] = df['is_listened'][i]
print(df_pivot_manual)

# print(df_pivot_manual.dtypes)
# print(df_pivot_manual.memory_usage())