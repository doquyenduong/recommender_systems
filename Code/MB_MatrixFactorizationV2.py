import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import tqdm
from deezerData import readData

# https://medium.com/analytics-vidhya/matrix-factorization-as-a-recommender-system-727ee64683f0
# https://wiki.ubc.ca/Course:CPSC522/Recommendation_System_using_Matrix_Factorization
def matrix_factorization(R, P, Q, K, steps=100, alpha=0.03, beta=0.2):
    error_list = []
    Q = Q.T
    for step in tqdm.tqdm(range(steps)):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > -1:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        error_list.append(e)
        if e < 0.001:
            break
    return P, Q.T , error_list


## Data import; first for coding train_short.csv; final version with train.csv
# dataframe must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order
# df, X, y, X_train, X_test, y_train, y_test
df_import, _, _, _, _, _, _ = readData()
print(df_import.info())
df = df_import.iloc[:, [11, 2, 14]]
# df = pd.read_csv('../Data/train_short.csv').iloc[:, [11, 2, 14]]
print(df.info(), df.shape)
df_r = pd.pivot_table(df, index=['user_id'], columns=['media_id'], values=['is_listened'], fill_value=-1).astype(int)
print(df_r.head())
matrix_r = df_r.to_numpy(dtype='int8')
print(matrix_r)


R = matrix_r
N = len(R)
M = len(R[0])
K = 50

P = np.random.rand(N, K)
Q = np.random.rand(M, K)

nP, nQ, error_list_MF = matrix_factorization(R, P, Q, K)
nR = np.dot(nP, nQ.T)

print(nR)
print(np.around(nR))
nR1 = np.clip(np.around(nR), a_min=0, a_max=1)
print(nR.shape)

# creating r_hat data frame
index_values = sorted(list(set(df['user_id'].sort_values(ascending=True))))
media_values = sorted(list(set(df['media_id'].sort_values(ascending=True))))

r_hat = pd.DataFrame(nR1, index=index_values, columns=media_values, dtype='int')
r_hat.index.name = 'user_id'
r_hat.columns.name = 'media_id'
print(r_hat.info(), r_hat.shape)
print(r_hat.head())
r_hat_unpivot = r_hat.stack().reset_index()
print(r_hat_unpivot)


# calculating roc_auc for the existing values
df_r2 = df_r.droplevel(level=0, axis=1).reset_index()
df_r_unpivoted = pd.melt(df_r2, id_vars=['user_id'], value_name='is_listened')
print(df_r2.shape, df_r2.head())
df_auc = df.merge(r_hat_unpivot, left_on=['user_id', 'media_id'], right_on=['user_id', 'media_id'])
print('auc for existing ratings',roc_auc_score(df_auc.iloc[:, 2], df_auc.iloc[:, 3]))

error_series = pd.Series(error_list_MF)
error_series.plot(kind='line')
plt.savefig("MatrixFactorizationError.png")
plt.show()

