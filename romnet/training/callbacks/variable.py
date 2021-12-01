import sys
import tensorflow.keras.callbacks as CB

from tensorflow.python.util.tf_export import keras_export


#=======================================================================================================================================
@keras_export('keras.callbacks.VariableValue')
class VariableValue(CB.Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variables.
        period (int): Interval (number of epochs) between checking values.
        filepath (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
    """

    def __init__(self, var_list, period=1, filepath=None):
        super(VariableValue, self).__init__()

        self.var_list  = var_list
        self.names     = [weight.name for layer in self.model.layers for weight in layer.weights]
        self.idx       = [self.names.index(i) for i in self.var_list]

        self.period    = period
        self.precision = precision

        self.file = sys.stdout if filepath is None \
                    else open(filepath+'/var_values.csv', "w", buffering=1)
        if self.file is not sys.stdout: self.file.write("#epoch",*self.var_list, sep=',', end='\n')

        self.values = None
        self.epochs_since_last = 0

    def on_train_begin(self, logs=None):
        self.values = [list(self.model.get_weights())[i] for i in self.idx]
        if self.file is not sys.stdout:
            self.file.write("0",*self.values, sep=',',end='\n')
            self.file.flush()
        utils.print_main("Variables values at epoch %05d:\n" % 0)
        for i, name in enumerate(self.var_list): print("  -", name, "=", self.values[i])

    def on_train_end(self, logs=None):
        if self.file is not sys.stdout:
            self.file.close()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.values = [list(self.model.get_weights())[i] for i in self.idx]
            if self.file is not sys.stdout:
                self.file.write(epoch,*self.values, sep=',',end='\n')
                self.file.flush()
            utils.print_main("Variables values at epoch %05d:\n" % epoch)
            for i, name in enumerate(self.var_list): print("  -", name, "=", self.values[i])

    def get_value(self):
        """Return the variable values."""
        return self.values

#=======================================================================================================================================
