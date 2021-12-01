import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as CB

from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils


@keras_export('keras.callbacks.LRPiecewiseConstantDecay')
class LRPiecewiseConstantDecay(CB.Callback):
    """A Learning Rate Schedule that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries, values, verbose=0):
        super(LRPiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            utils.raise_value_err(
                "In `{}` callback, the length of `boundaries` should "
                "be 1 less than the length of `values`.".format(self.__name__)
            )
        self.boundaries = np.array(boundaries)
        self.values     = np.array(values)
        self.idx        = 0

    def on_epoch_end(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            utils.raise_value_err("Optimizer must have a `lr` attribute.")
        greater = self.boundaries < epoch
        if True in greater:
            idx = np.where(greater)[0][-1]
            self.idx = idx+1
            new_lr = self.values[self.idx]
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                utils.print_main(
                    "Epoch %05d: LRPiecewiseConstantDecay reducing "
                    "learning rate to %s." % (epoch + 1, new_lr)
                )


@keras_export('keras.callbacks.LearningRateTracker')
class LearningRateTracker(CB.Callback):
    """Learning rate tracker."""

    def __init__(
        self,
        file=None,
        log_dir=None,
        verbose=1
    ):
        super(LearningRateTracker, self).__init__()

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_lr = tf.summary.create_file_writer(
                log_dir + '/scalars/lr'
            )

        self.file = None if file is None else open(file, "w", buffering=1)
        if self.file is not None:
            self.file.write("lr\n")

        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        lr = float(K.get_value(self.model.optimizer._decayed_lr(tf.float64)))
        if 'learning_rate' not in logs:
            logs['learning_rate'] = lr
        if self.log_dir is not None:
            with self.log_lr.as_default():
                tf.summary.scalar(
                    name='epoch_learning_rate', data=lr, step=epoch
                )
        if self.file is not None:
            self.file.write('%.6e\n' % lr)
            self.file.flush()
        if self.verbose > 0:
            print("Learning rate value = %.4e" % lr)

    def on_train_end(self, logs=None):
        if self.file is not None:
            self.file.close()


