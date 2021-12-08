import time
import tensorflow                         as tf
import pandas                             as pd

from tensorflow.python.util.tf_export import keras_export
import tensorflow.keras.callbacks         as CB


#=======================================================================================================================================
@keras_export('keras.callbacks.BaseLogger')
class BaseLogger(CB.Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    Args:
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is in `on_epoch_end`.
          All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, stateful_metrics=None, PathToRunFld=None):
        super(BaseLogger, self).__init__()
        self.stateful_metrics = set(stateful_metrics or [])
        self.PathToRunFld     = PathToRunFld
        self.first_time       = True


    def on_epoch_begin(self, epoch, logs=None):
        self.seen       = 0
        self.totals     = {}



    def on_batch_end(self, batch, logs=None):
        
        logs             = logs or {}
        batch_size       = logs.get('size', 1)
        num_steps        = logs.get('num_steps', 1)
        
        if (self.seen == 0):
            keys             = list(logs.keys())
            self.all_metrics = [keys[i] for i in range(len(keys)) if ('loss' in keys[i])]

        self.seen += batch_size * num_steps

        for k, v in logs.items():
            if k in self.all_metrics:
                if k in self.stateful_metrics:
                    self.totals[k] = v
                else:
                    if k in self.totals:
                        self.totals[k] += v * batch_size
                    else:
                        self.totals[k]  = v * batch_size



    def on_epoch_end(self, epoch, logs=None):

        if logs is not None:
            for k in self.all_metrics:
                if k in self.stateful_metrics:
                    logs[k] = self.totals[k]
                else:
                    logs[k] = self.totals[k] / self.seen

        logss = {}
        for key, val in logs.items():
            logss[key] = [val]
        if (self.first_time):
            self.first_time = False
            pd.DataFrame.from_dict(logss).to_csv(path_or_buf=self.PathToRunFld+'/Training/History.csv', index=False, mode='w', header=True)
        else:
            pd.DataFrame.from_dict(logss).to_csv(path_or_buf=self.PathToRunFld+'/Training/History.csv', index=False, mode='a', header=False)

#=======================================================================================================================================
