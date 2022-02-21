#!/usr/bin/env python3

## Code was slightly adapted from https://github.com/keras-team/keras/issues/6050.
## Needed to create batch-wise quality metrics. For articles related to this issue, see also:
# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
# https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras


from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class RocCallback(Callback):

    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train, 4)),
                                                         str(round(roc_val, 4))), end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
