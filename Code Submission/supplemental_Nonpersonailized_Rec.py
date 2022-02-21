# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:17:47 2022
"""
# Non-personalized recommendations

## One of the most basic ways to make recommendations is to go with the knowledge 
## of the crowd and recommend what is already the most popular.
## Under this part, we will combine the two factors: the songs appears the most 
## from the dataset and the songs with the highest listening frequency

## This part is not the main focus of our study but to explore different types of recommendations

## Import libraries
import pandas as pd
import numpy as np

## Read file
df = pd.read_csv("../Data/train.csv")

#----------------------------------------------------------------------
# Extract relevant columns
#----------------------------------------------------------------------
df_extracted = df[["user_id", "media_id", "is_listened"]]

# Get the counts of occurrences of each song id
df_song_count = df_extracted['media_id'].value_counts()
df_song_count.head()
# Number of unique counts
df_song_count.unique()

#----------------------------------------------------------------------
# Find the sum of the is_listened given to each song id
#----------------------------------------------------------------------
# Using the sum of is_listened because the more times a song was listened, the more popular it is
avg_listened_df = df_extracted[["media_id", "is_listened"]].groupby(['media_id']).sum()
avg_listened_df.head()
sorted_avg_listened_df = avg_listened_df.sort_values(by="is_listened", ascending=False)
sorted_avg_listened_df.head()

#----------------------------------------------------------------------
# Create a list of only songs appearing > 10000 times in the dataset
#----------------------------------------------------------------------
frequently_listened_songs = df_song_count[df_song_count > 10000].index

#----------------------------------------------------------------------
# Use this frequently_listened_songs list to filter the original DataFrame
#----------------------------------------------------------------------
df_frequent_songs = df_extracted[df_extracted["media_id"].isin(frequently_listened_songs)]
frequent_song_avgs = df_frequent_songs[["media_id", "is_listened"]].groupby('media_id').sum()
# Inspect the songs with their listening frequency
print(frequent_song_avgs.sort_values(by="is_listened", ascending=False).head())
