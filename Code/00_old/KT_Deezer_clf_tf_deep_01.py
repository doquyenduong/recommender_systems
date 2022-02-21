#!/usr/bin/env python3

## similar to mf_keras_deep.py (Day4, part2)

from deezerData import readData, extractValuesToVec
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.optimizers import SGD
from keras.losses import BinaryCrossentropy


## get datasets
df_train, df_test, arr_train, arr_test, test_csv_boole = readData()

## get vectors for users, songs and if the song was listened to (binary, threshold 30s)
vec_train_userID, vec_train_songID, vec_test_userID, vec_test_songID, vec_train_isListened, vec_test_isListened = \
  extractValuesToVec(df_train, df_test, test_csv_boole)

## define number variables
N = max(vec_train_userID.max(), vec_test_userID.max()) + 1  ## max of userID
M = max(vec_train_songID.max(), vec_test_songID.max()) + 1  ## max of songID


## initialize variables
reg = 0.0001   ## regularization penalty
lr = 0.08      ## learning rate
K = 10         ## latent dimensionality
mu = vec_train_isListened.mean()
epochs = 50

## keras model
u = Input(shape=(1,))
m = Input(shape=(1,))
u_embedding = Embedding(N, K)(u)   ## (N, 1, K)
m_embedding = Embedding(M, K)(m)   ## (N, 1, K)
u_embedding = Flatten()(u_embedding)  ## (N, K)
m_embedding = Flatten()(m_embedding)  ## (N, K)
x = Concatenate()([u_embedding, m_embedding])  ## (N, 2K)

## the neural network
x = Dense(400)(x)
# x = BatchNormalization()(x)
x = Activation('relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(100)(x)
# x = BatchNormalization()(x)
# x = Activation('sigmoid')(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss=BinaryCrossentropy(from_logits=True),
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['AUC'],
)

r = model.fit(
  x=[vec_train_userID, vec_train_songID],
  y=vec_train_isListened - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [vec_test_userID, vec_test_songID],
    vec_test_isListened - mu
  )
)


# print(r.history.keys())
#
plt.figure()

## plot loss
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("model loss")
plt.show()

## plot AUC
plt.plot(r.history['auc'], label="train AUC")
plt.plot(r.history['val_auc'], label="test AUC")
plt.xlabel("epoch")
plt.ylabel("AUC")
plt.title("model AUC")
plt.legend()
plt.show()

# ## one plot for everything
# pd.DataFrame(r.history).plot(figsize=(8,5))
# plt.xlabel("epoch")
# plt.ylabel("metric")
# plt.title("model metric")
# plt.show()