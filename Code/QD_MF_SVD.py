# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:48:08 2022

"""
# Singular value decomposition (SVD)
# The codes are written according to the DataCamp course
# There are no user ratings in this dataset. Thus, we use  

## Import libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error

## Read file
# Sample file
from deezerData import readData
df, X, y, X_train, X_test, y_train, y_test = readData()

#----------------------------------------------------------------------
# Create a dataframe with variables of interest 
#----------------------------------------------------------------------
# First selecting variables, then group by user_id -> it helps to pivot and set user_id as index ! 
df_extracted = df[["user_id", "media_id", "is_listened"]]
df_extracted = df_extracted.groupby(["user_id", "media_id"], as_index=False)["is_listened"].sum()
df_extracted = df_extracted.sort_values(by = ["user_id", "media_id"], axis=0)

# Pivot the dataframe and make each media_id as a column
df_extracted_pivot = df_extracted.pivot(index="user_id", 
                                        columns="media_id", 
                                        values="is_listened").fillna(0)

# Create a matrix from the pivoted dataframe
m_extracted = df_extracted_pivot.values

#----------------------------------------------------------------------
# Normalize the data by de-meaning / centering isListened
#----------------------------------------------------------------------
# Before we can find the factors of the ratings matrix using singular value decomposition, 
# we will need to "de-mean", or center it, by subtracting each row's mean from each value in that row.

# Get the average is_listened for each user, then reshape 
avg_islistened = np.mean(m_extracted, axis=1)
avg_islistened_reshape = avg_islistened.reshape(-1,1)

# Center each user's is_listened around 0
m_isListened_centered = m_extracted - avg_islistened_reshape

#----------------------------------------------------------------------
# Decompose the matrix
#----------------------------------------------------------------------
U, sigma, Vt = svds(m_isListened_centered)
# Dot product of U and sigma
U_sigma = np.dot(U, sigma)
sigma = np.diag(sigma)
print(sigma)
U_sigma_V = np.dot(np.dot(U, sigma), Vt)

#----------------------------------------------------------------------
# Validating predictions
#----------------------------------------------------------------------
# Predicting by adding the average back
all_user_pred_isListened = U_sigma_V + avg_islistened_reshape
df_pred = pd.DataFrame(all_user_pred_isListened, 
                       columns=df_extracted_pivot.columns,
                       index=df_extracted_pivot.index
                       )

# Extract the ground truth to compare predictions against
actual_values = df_extracted_pivot.iloc[:20, :100].values
predicted_values = df_pred.iloc[:20, :100].values

# Create a mask of actual_values to only look at the non-missing values in the ground truth
mask = ~np.isnan(actual_values)

# Print the performance of both predictions and compare
print(mean_squared_error(actual_values[mask], predicted_values[mask], squared=False))

#----------------------------------------------------------------------
# Predicting on 1 user
#----------------------------------------------------------------------
# Sort the score for songs of user 5 from high to low
user_5_song = df_pred.loc[5,:].sort_values(ascending=False)
print(user_5_song)
