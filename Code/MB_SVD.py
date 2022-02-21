import pandas as pd
from surprise import accuracy, Dataset, Reader, SVD, SVDpp
from surprise.model_selection import cross_validate, train_test_split
import time
from deezerData import readData
start_time = time.time()

## Data import; first for coding train_short.csv; final version with train.csv
# dataframe must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order
df_import, _, _, _, _, _, _ = readData()

df = df_import.iloc[:, [11, 2, 14]]

reader = Reader(rating_scale=(0, 1))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['user_id', 'media_id', 'is_listened']], reader)

# Defining the algorithm
algo_svd = SVD()
algo_svdpp = SVDpp()

# # Run 5-fold cross-validation and print results
# print('\nSVD crossvalidation')
# cross_validate(algo_svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# print('\nSVD++ crossvalidation')
# cross_validate(algo_svdpp, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

load_time = time.time()
total_load_time = load_time - start_time
print('\nTotal time for loading the data', total_load_time)

# sample random trainset and testset
# test set is made of 25% of the ratings.
print('\nSplitting the data')
trainset, testset = train_test_split(data, test_size=.25)

tts_time = time.time()
total_tts_time = tts_time - load_time
print('Total time for splitting the data', total_tts_time)

############## SVD ##########################
# Train the algorithm on the trainset, and predict ratings for the testset
print('\nFitting SVD')
algo_svd.fit(trainset)

fitsvd_time = time.time()
total_fitsvd_time = fitsvd_time - tts_time
print('Total time for splitting the data', total_fitsvd_time)

print('\nPredicting SVD')
predictions_svd = algo_svd.test(testset)

predsvd_time = time.time()
total_predsvd_time = predsvd_time - fitsvd_time
print('Total time for splitting the data', total_predsvd_time)

# Then compute RMSE
print('\nSVD RMSE')
accuracy.rmse(predictions_svd)

svd_time = time.time()
total_svd_time = svd_time - tts_time
print('Total time for SVD', total_svd_time)

############## SVD++ ##########################
# Train the algorithm on the trainset, and predict ratings for the testset
print('\nFitting SVD++')
algo_svdpp.fit(trainset)

fitsvdpp_time = time.time()
total_fitsvdpp_time = fitsvdpp_time - svd_time
print('Total time for fitting SVD++', total_fitsvdpp_time)

print('\nPredicting SVD++')
predictions_svdpp = algo_svdpp.test(testset)

predsvdpp_time = time.time()
total_predsvdpp_time = predsvdpp_time - fitsvdpp_time
print('Total time for predicting SVD++', total_predsvdpp_time)

# Then compute RMSE
print('\nSVD++ RMSE')
accuracy.rmse(predictions_svdpp)

svdpp_time = time.time()
total_svdpp_time = svdpp_time - svd_time
print('Total time for SVD++', total_svdpp_time)

end_time = time.time()
total_time = end_time - start_time
print('Total time elapsed', total_time)