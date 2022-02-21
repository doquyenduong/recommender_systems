import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import statsmodels.api as sm

## save a smaller csv of the training data
################################################################
# df = pd.read_csv('./Data/train.csv')
#
# df_short = df.iloc[0:1000,:]
#
# df_short.to_csv('./Data/train_short.csv', index=False)
################################################################

## Data import; first for coding train_short.csv; final version with train.csv
df = pd.read_csv("../Data/train_short.csv")

# Define data frames for input variables and response variable
df_X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
df_y = df.iloc[:, -1:]

# Divide it into train and test data frames
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25, stratify=df_y, random_state=42)

# Fit the logistic regression model
log_reg = sm.Logit(y_train, X_train).fit()

# printing the summary table
print(log_reg.summary())

##### calculating metrics for train data #####
# performing predictions on the train data set
x_hat = log_reg.predict(X_train)
prediction = list(map(round, x_hat))

# confusion matrix
cm = confusion_matrix(y_train, prediction)
print ("Confusion Matrix : \n", cm)

# accuracy score of the model
print('Test accuracy = ', accuracy_score(y_train, prediction))

auc_train = roc_auc_score(y_train, prediction)
print ("AuC : \n", auc_train)

##### calculating metrics for test data #####
# Actual prediction
# performing predictions on the test data set
y_hat = log_reg.predict(X_test)
prediction_test = list(map(round, y_hat))

# confusion matrix
cm = confusion_matrix(y_test, prediction_test)
print ("Confusion Matrix : \n", cm)

# accuracy score of the model
print('Test accuracy = ', accuracy_score(y_test, prediction_test))

#### Calculation/Prediction of is_listened on test.csv
df_test = pd.read_csv('../Data/test.csv').iloc[:, 1:]
print(df_test.info())

y_hat_test = log_reg.predict(df_test)
prediction_test_final = list(map(round, y_hat_test))
print(prediction_test_final)
