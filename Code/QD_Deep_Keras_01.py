# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:31:44 2022
"""
# Deep Learning with Keras
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Read file
df = pd.read_csv("../Data/train.csv")
df["sample_id"] = df.index
### Make a sample 
# df = df.sample(frac=0.005)             #### with 0.5% of original dataset
df = df[df["user_id"] < 200]
df = df[df["media_id"] < 400000]

# Use pairplot and set the hue to be the class column
sns.pairplot(df, hue='is_listened') 
# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', df.describe())

# Count the number of observations per class
print('Observations per class: \n', df['is_listened'].value_counts())

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
# Split the dataset into its attributes and labels -> make arrays
X = df.drop('is_listened', axis=1).values
y = df["is_listened"].values

# Train Test Split: 80% train data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=123)

# ----------------------------------------------------------------------
# Simple model
# ----------------------------------------------------------------------
# Create a sequential model
model = Sequential()

# Add input and hidden layer
model.add(Dense(10, input_shape=(15,), activation="tanh"))
model.add(Dense(10, activation="tanh"))

# Add output layer, use sigmoid
model.add(Dense(1, activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

# Display a summary of your model
model.summary()

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs = 20, validation_split=0.2)

model.predict(X_test)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


# ----------------------------------------------------------------------
# The history callback
# ----------------------------------------------------------------------
# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs = 50,
               validation_data=(X_test, y_test), verbose=0)

# Write the function to plot the loss & accuracy
def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

def plot_accuracy(accuracy,val_accuracy):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(accuracy)
  plt.plot(val_accuracy)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['accuracy'], h_callback.history['val_accuracy'])

# ----------------------------------------------------------------------
# A combination of callbacks
# ----------------------------------------------------------------------
# Deep learning models can take a long time to train, especially when we move to
# deeper architectures and bigger datasets. Saving our model every time it 
# improves as well as stopping it when it no longer does allows us to worry less 
# about choosing the number of epochs to train for. We can also restore a saved 
# model anytime and resume training where we left it.

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_accuracy', patience = 5)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit the model for a large amount of epochs
h_callback_1 = model.fit(X_train, y_train,
                    epochs = 1000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))
