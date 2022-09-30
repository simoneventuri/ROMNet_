import numpy as np
import os 
import h5py
import sys

import tensorflow_probability                   as tfp

import tensorflow                               as tf
from tensorflow.python.keras                import metrics as metrics_mod
from tensorflow                             import train

from tensorflow.python.keras                import backend
from tensorflow.python.keras.saving         import saving_utils
from tensorflow.python.training.tracking    import util as trackable_utils
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.engine         import data_adapter
from tensorflow.python.keras.utils          import losses_utils

from keras.utils                            import traceback_utils
from keras.engine                           import base_layer
from keras.engine                           import base_layer_utils

from ..utils                                import hdf5_format
  
from ..training                             import losscontainer



# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None
# pylint: enable=g-import-not-at-top



#=======================================================================================================================================
def load_model_(ModelFld):

    Architecture = tf.keras.Model.load_model(ModelFld)

    return Architecture

#=======================================================================================================================================



#=======================================================================================================================================
def load_weights_(ModelFld):

    # ModelFile         = ModelFld + '/MyModel/'
    # Architecture      = tf.keras.Model.load_model(ModelFile)
    # MCFile            = ModelFld + '/Params/ModelCheckpoint/cp-{epoch:04d}.ckpt'
    # checkpoint_dir    = os.path.dirname(MCFile)
    # latest            = train.latest_checkpoint(checkpoint_dir)

    ModelFld = ModelFld + "/Training/Params/"
    last = max(os.listdir(ModelFld), key=lambda x: int(x.split('.')[0]))
    if last:
        ModelFld = ModelFld + "/" + last
    print('\n[ROMNet - architecture.py                     ]:   Loading ML Model Parameters from File: ', ModelFld)

    Architecture.load_weights(ModelFld)

    return Architecture

#=======================================================================================================================================



#=======================================================================================================================================
@tf.keras.utils.register_keras_serializable(package='ROMNet', name='Architecture')
class Architecture(tf.keras.Model):
    """Base class for all surrogate modules."""

    def __init__(self):
        super(Architecture, self).__init__()

        self.sructure_name  = 'Architecture'

        self.attention_mask = None
        self.residual       = None


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'inp_trans':        self.inp_trans,
            'out_trans':        self.out_trans,
            'pde_loss_weights': self.pde_loss_weights,
            'residual':         self.residual,
            'data_ids':         self.data_ids,
            'data_ids_valid':   self.data_ids_valid
        }
        base_config = super(Architecture, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    @classmethod
    def from_config(cls, config):
        return cls(**config)


    @traceback_utils.filter_traceback
    #=======================================================================================================================================
    def compile(self,
                data,
                optimizer           = 'rmsprop',
                loss                = None,
                metrics             = None,
                loss_weights        = None,
                weighted_metrics    = None,
                run_eagerly         = None,
                steps_per_execution = None,
                jit_compile         = False,
                **kwargs):
        """Configures the model for training.
        Example:
        ```python
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                    loss=tf.keras.losses.BinaryCrossentropy(),
                                    metrics=[tf.keras.metrics.BinaryAccuracy(),
                                                     tf.keras.metrics.FalseNegatives()])
        ```
        Args:
                optimizer: String (name of optimizer) or optimizer instance. See
                    `tf.keras.optimizers`.
                loss: Loss function. Maybe be a string (name of loss function), or
                    a `tf.keras.losses.Loss` instance. See `tf.keras.losses`. A loss
                    function is any callable with the signature `loss = fn(y_true,
                    y_pred)`, where `y_true` are the ground truth values, and
                    `y_pred` are the model's predictions.
                    `y_true` should have shape
                    `(batch_size, d0, .. dN)` (except in the case of
                    sparse loss functions such as
                    sparse categorical crossentropy which expects integer arrays of shape
                    `(batch_size, d0, .. dN-1)`).
                    `y_pred` should have shape `(batch_size, d0, .. dN)`.
                    The loss function should return a float tensor.
                    If a custom `Loss` instance is
                    used and reduction is set to `None`, return value has shape
                    `(batch_size, d0, .. dN-1)` i.e. per-sample or per-timestep loss
                    values; otherwise, it is a scalar. If the model has multiple outputs,
                    you can use a different loss on each output by passing a dictionary
                    or a list of losses. The loss value that will be minimized by the
                    model will then be the sum of all individual losses, unless
                    `loss_weights` is specified.
                metrics: List of metrics to be evaluated by the model during training
                    and testing. Each of this can be a string (name of a built-in
                    function), function or a `tf.keras.metrics.Metric` instance. See
                    `tf.keras.metrics`. Typically you will use `metrics=['accuracy']`. A
                    function is any callable with the signature `result = fn(y_true,
                    y_pred)`. To specify different metrics for different outputs of a
                    multi-output model, you could also pass a dictionary, such as
                    `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                    You can also pass a list to specify a metric or a list of metrics
                    for each output, such as `metrics=[['accuracy'], ['accuracy', 'mse']]`
                    or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
                    strings 'accuracy' or 'acc', we convert this to one of
                    `tf.keras.metrics.BinaryAccuracy`,
                    `tf.keras.metrics.CategoricalAccuracy`,
                    `tf.keras.metrics.SparseCategoricalAccuracy` based on the loss
                    function used and the model output shape. We do a similar
                    conversion for the strings 'crossentropy' and 'ce' as well.
                loss_weights: Optional list or dictionary specifying scalar coefficients
                    (Python floats) to weight the loss contributions of different model
                    outputs. The loss value that will be minimized by the model will then
                    be the *weighted sum* of all individual losses, weighted by the
                    `loss_weights` coefficients.
                        If a list, it is expected to have a 1:1 mapping to the model's
                            outputs. If a dict, it is expected to map output names (strings)
                            to scalar coefficients.
                weighted_metrics: List of metrics to be evaluated and weighted by
                    `sample_weight` or `class_weight` during training and testing.
                run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s
                    logic will not be wrapped in a `tf.function`. Recommended to leave
                    this as `None` unless your `Model` cannot be run inside a
                    `tf.function`. `run_eagerly=True` is not supported when using
                    `tf.distribute.experimental.ParameterServerStrategy`.
                steps_per_execution: Int. Defaults to 1. The number of batches to run
                    during each `tf.function` call. Running multiple batches inside a
                    single `tf.function` call can greatly improve performance on TPUs or
                    small models with a large Python overhead. At most, one full epoch
                    will be run each execution. If a number larger than the size of the
                    epoch is passed, the execution will be truncated to the size of the
                    epoch. Note that if `steps_per_execution` is set to `N`,
                    `Callback.on_batch_begin` and `Callback.on_batch_end` methods will
                    only be called every `N` batches (i.e. before/after each `tf.function`
                    execution).
                jit_compile: If `True`, compile the model training step with XLA.
                    [XLA](https://www.tensorflow.org/xla) is an optimizing compiler for
                    machine learning.
                    `jit_compile` is not enabled for by default.
                    This option cannot be enabled with `run_eagerly=True`.
                    Note that `jit_compile=True` is
                    may not necessarily work for all models.
                    For more information on supported operations please refer to the
                    [XLA documentation](https://www.tensorflow.org/xla).
                    Also refer to
                    [known XLA issues](https://www.tensorflow.org/xla/known_issues) for
                    more details.
                **kwargs: Arguments supported for backwards compatibility only.
        """
        base_layer.keras_api_gauge.get_cell('compile').set(True)
        with self.distribute_strategy.scope():
            if 'experimental_steps_per_execution' in kwargs:
                logging.warning('The argument `steps_per_execution` is no longer '
                                                'experimental. Pass `steps_per_execution` instead of '
                                                '`experimental_steps_per_execution`.')
                if not steps_per_execution:
                    steps_per_execution = kwargs.pop('experimental_steps_per_execution')

            # When compiling from an already-serialized model, we do not want to
            # reapply some processing steps (e.g. metric renaming for multi-output
            # models, which have prefixes added for each corresponding output name).
            from_serialized = kwargs.pop('from_serialized', False)


            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            self.optimizer = self._get_optimizer(optimizer)

            # -------------------------------------------------------------------------------------------------------------------------> Changed TF
            # Defining loss containers
            self.compiled_loss = {}
            for data_id in self.data_ids:
                _loss                                    = loss[data_id] if loss else None
                _loss_weights                            = loss_weights[data_id] if loss_weights else None
                self.compiled_loss[data_id]              = losscontainer.LossesContainer(_loss, loss_weights=_loss_weights, output_names=self.output_vars)
                self.compiled_loss[data_id]._loss_metric = metrics_mod.Mean(name=data_id + '_loss', dtype=self.dtype)
            
            # Defining metrics container
            if metrics is not None:
                print( "[ROMNet - architecture.py                     ]   WARNING! No Metrics Implemented." )
                # self.compiled_metrics = compile_utils.MetricsContainer(metrics, weighted_metrics, output_names=self.output_names, from_serialized=from_serialized)
            self.compiled_metrics = None
            # -------------------------------------------------------------------------------------------------------------------------> Changed TF

            self._configure_steps_per_execution(steps_per_execution or 1)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True
            self.loss = loss or {}
            if (self._run_eagerly or self.dynamic):
                self._jit_compile = False
            else:
                self._jit_compile = jit_compile

    #=======================================================================================================================================



    #=======================================================================================================================================
    @property
    def metrics(self):

        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                for val in self.compiled_loss.values():
                    metrics += [val._loss_metric]

        for l in self._flatten_layers():
            metrics.extend(l._metrics)  # pylint: disable=protected-access
        
        return metrics

    #=======================================================================================================================================



    #=======================================================================================================================================
    def train_step(self, data):
        """Custom train step."""

        x, y, indx = data_adapter.expand_1d(data)

        # Gradient descent step
        losses = {}
        with tf.GradientTape() as tape:
            
            for i, data_id in enumerate(self.data_ids_valid):
                losses[data_id] = self._get_loss(i, x[i], y[i], tf.squeeze(indx[i]), data_id, training=True)

            # Calculate total loss
            total_loss = tf.add_n(losses_utils.cast_losses_to_common_dtype(losses.values()))


            # Add regularization losses
            if self.losses:
                reg_loss    = tf.add_n(losses_utils.cast_losses_to_common_dtype(self.losses))
                total_loss += reg_loss


        # Update parameters
        self.optimizer.minimize( total_loss, self.trainable_variables, tape=tape )


        # Collect metrics
        metrics = {m.name: m.result() for m in self.metrics}
        if reg_loss is not None:
            metrics.update({'reg_loss': reg_loss})

        if self.pde_loss_weights is None:
            metric = {'tot_loss': tf.add_n(metrics.values())}
        else:
            metric = {'tot_loss': tf.add_n(metrics.values())}        

        # if self.pde_loss_weights is None:
        #     metric = {}
        # else:
        #     # Add weighted total loss
        #     metric = {'tot_loss_weighted': tot_loss_weighted}
        # metrics = _get_metrics(tot_loss, losses, reg_loss=reg_loss)

        return {**metric, **metrics}



    #=======================================================================================================================================



    #=======================================================================================================================================
    def test_step(self, data):
        """Custom test step."""

        x, y, indx = data_adapter.expand_1d(data)

        losses = {}
        for i, data_id in enumerate(self.data_ids_valid):
            losses[data_id] = self._get_loss(i, x[i], y[i], indx[i], data_id, training=False)


        # Collect metrics
        metrics = {m.name: m.result() for m in self.metrics}
        if self.pde_loss_weights is None:
            metric = {'tot_loss': tf.add_n(metrics.values())}
        else:
            metric = {'tot_loss': tf.add_n(metrics.values())}         


        # # Calculate total loss
        # total_loss = tf.add_n(losses_utils.cast_losses_to_common_dtype(losses.values()))

        # metrics = _get_metrics_test(total_loss, losses)

        return {**metric, **metrics}

    #=======================================================================================================================================



    #=======================================================================================================================================
    def _get_loss(self, i, x, y, indx, data_id, training=True):
                
        if data_id == 'res':
            y_pred = self.residual(x, training=training)
            if isinstance(y_pred, (list,tuple)):
                y = [ tf.zeros_like(y_pred_i) for y_pred_i in y_pred ]
            else:
                y = tf.zeros_like(y_pred)
        else:
            y_pred = self(x, training=training)
            if (self.fROM_anti):
                y_pred = self.fROM_anti(y_pred)

        if ((self.attention_mask is None) or (data_id == 'pts') or (not training)):
            attention_mask = None
        else:
            attention_mask = tf.gather(self.attention_mask[i], indx, axis=0)         
        
        if (self.pde_loss_weights):
            loss_ = self.compiled_loss[data_id](y, y_pred, attention_mask=attention_mask) * self.pde_loss_weights[data_id]
        else:
            loss_ = self.compiled_loss[data_id](y, y_pred, attention_mask=attention_mask)

        return loss_

    #=======================================================================================================================================



    #=======================================================================================================================================
    def _get_tot_loss(self, losses, reg_loss):
        
        total_loss          = reg_loss 
        total_loss_weighted = reg_loss 

        for data_id, loss in losses.items():

            if (self.pde_loss_weights):
                total_loss_weighted += loss * self.pde_loss_weights[data_id] 
            else:
                total_loss_weighted += loss 
            
            total_loss += loss 

        return total_loss, total_loss_weighted

    #=======================================================================================================================================



    #=======================================================================================================================================
    def _get_tot_loss_test(self, losses):
        
        total_loss = 0.
        for data_id, loss in losses.items():

            # if self.pde_loss_weights is not None:
            #     total_loss += self.pde_loss_weights[data_id] * loss 
            # else:
            #     total_loss += loss 

            total_loss += loss 

        return total_loss

    #=======================================================================================================================================


    # Input/Output transformations
    #=======================================================================================================================================
    @property
    def inp_trans(self):
        return self._inp_trans
    #=======================================================================================================================================



    #=======================================================================================================================================
    @inp_trans.setter
    def inp_trans(self, function):
        self._inp_trans = function
    #=======================================================================================================================================



    #=======================================================================================================================================
    @property
    def out_trans(self):
        return self._out_trans
    #=======================================================================================================================================



    #=======================================================================================================================================
    @out_trans.setter
    def out_trans(self, function):
        self._out_trans = function
    #=======================================================================================================================================



    # Data identities
    #=======================================================================================================================================
    @property
    def data_ids(self):
        return self._data_ids
    #=======================================================================================================================================



    #=======================================================================================================================================
    @data_ids.setter
    def data_ids(self, identifiers):
        self._data_ids = identifiers
    #=======================================================================================================================================



    #=======================================================================================================================================
    @property
    def data_ids_valid(self):
        return self._data_ids_valid
    #=======================================================================================================================================



    #=======================================================================================================================================
    @data_ids_valid.setter
    def data_ids_valid(self, identifiers):
        self._data_ids_valid = identifiers
    #=======================================================================================================================================



    # Losses properties
    #=======================================================================================================================================
    @property
    def pde_loss_weights(self):
        return self._pde_loss_weights
    #=======================================================================================================================================



    #=======================================================================================================================================
    @pde_loss_weights.setter
    def pde_loss_weights(self, weights):
        self._pde_loss_weights = weights
    #=======================================================================================================================================



    #=======================================================================================================================================
    # Residual loss
    @property
    def residual(self):
        return self._residual
    #=======================================================================================================================================



    #=======================================================================================================================================
    @residual.setter
    def residual(self, function):
        self._residual = function

    #=======================================================================================================================================


    #=======================================================================================================================================
    def load_weights(self,
                     filepath,
                     by_name=False,
                     skip_mismatch=False,
                     options=None):
        """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.
        If `by_name` is False weights are loaded based on the network's
        topology. This means the architecture should be the same as when the weights
        were saved.  Note that layers that don't have weights are not taken into
        account in the topological ordering, so adding or removing layers is fine as
        long as they don't have weights.
        If `by_name` is True, weights are loaded into layers only if they share the
        same name. This is useful for fine-tuning or transfer-learning models where
        some of the layers have changed.
        Only topological loading (`by_name=False`) is supported when loading weights
        from the TensorFlow format. Note that topological loading differs slightly
        between TensorFlow and HDF5 formats for user-defined classes inheriting from
        `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
        TensorFlow format loads based on the object-local names of attributes to
        which layers are assigned in the `Model`'s constructor.
        Args:
                filepath: String, path to the weights file to load. For weight files in
                        TensorFlow format, this is the file prefix (the same as was passed
                        to `save_weights`). This can also be a path to a SavedModel
                        saved from `model.save`.
                by_name: Boolean, whether to load weights by name or by topological
                        order. Only topological loading is supported for weight files in
                        TensorFlow format.
                skip_mismatch: Boolean, whether to skip loading of layers where there is
                        a mismatch in the number of weights, or a mismatch in the shape of
                        the weight (only valid when `by_name=True`).
                options: Optional `tf.train.CheckpointOptions` object that specifies
                        options for loading weights.
        Returns:
                When loading a weight file in TensorFlow format, returns the same status
                object as `tf.train.Checkpoint.restore`. When graph building, restore
                ops are run automatically as soon as the network is built (on first call
                for user-defined classes inheriting from `Model`, immediately if it is
                already built).
                When loading weights in HDF5 format, returns `None`.
        Raises:
                ImportError: If h5py is not available and the weight file is in HDF5
                        format.
                ValueError: If `skip_mismatch` is set to `True` when `by_name` is
                    `False`.
        """

        #=======================================================================================================================================
        def _detect_save_format(filepath):
            """Returns path to weights file and save format."""

            filepath = path_to_string(filepath)
            if saving_utils.is_hdf5_filepath(filepath):
                return filepath, 'h5'

            # Filepath could be a TensorFlow checkpoint file prefix or SavedModel
            # directory. It's possible for filepath to be both a prefix and directory.
            # Prioritize checkpoint over SavedModel.
            if _is_readable_tf_checkpoint(filepath):
                save_format = 'tf'
            elif sm_loader.contains_saved_model(filepath):
                ckpt_path = os.path.join(filepath, sm_constants.VARIABLES_DIRECTORY,
                                                                 sm_constants.VARIABLES_FILENAME)
                if _is_readable_tf_checkpoint(ckpt_path):
                    filepath = ckpt_path
                    save_format = 'tf'
                else:
                    raise ValueError('Unable to load weights. filepath {} appears to be a SavedModel directory, but checkpoint either doesn\'t exist, or is incorrectly formatted.'.format(filepath))
            else:
                # Not a TensorFlow checkpoint. This filepath is likely an H5 file that
                # doesn't have the hdf5/keras extensions.
                save_format = 'h5'
            return filepath, save_format

        #=======================================================================================================================================


        if backend.is_tpu_strategy(self._distribution_strategy):
            if (self._distribution_strategy.extended.steps_per_run > 1 and
                    (not saving_utils.is_hdf5_filepath(filepath))):
                raise ValueError('Load weights is not yet supported with TPUStrategy with steps_per_run greater than 1.')
        if skip_mismatch and not by_name:
            raise ValueError(
                    'When calling model.load_weights, skip_mismatch can only be set to True when by_name is True.')

        filepath, save_format = _detect_save_format(filepath)
        if save_format == 'tf':
            status = self._trackable_saver.restore(filepath, options)
            if by_name:
                raise NotImplementedError(
                        'Weights may only be loaded based on topology into Models when loading TensorFlow-formatted weights (got by_name=True to load_weights).')
            if not context.executing_eagerly():
                session = backend.get_session()
                # Restore existing variables (if any) immediately, and set up a
                # streaming restore for any variables created in the future.
                trackable_utils.streaming_restore(status=status, session=session)
            status.assert_nontrivial_match()
        else:
            status = None
            if h5py is None:
                raise ImportError('`load_weights` requires h5py when loading weights from HDF5.')
            if not self._is_graph_network and not self.built:
                raise ValueError('Unable to load weights saved in HDF5 format into a subclassed Model which has not created its variables yet. Call the Model first, then load the weights.')
            self._assert_weights_created()
            with h5py.File(filepath, 'r') as f:
                if 'layer_names' not in f.attrs and 'model_weights' in f:
                    f = f['model_weights']
                if by_name:
                    hdf5_format.load_weights_from_hdf5_group_by_name(f, self.layers, skip_mismatch=skip_mismatch)
                else:
                    hdf5_format.load_weights_from_hdf5_group(f, self.layers)

        # Perform any layer defined finalization of the layer state.
        for layer in self.layers:
            layer.finalize_state()
    
    #=======================================================================================================================================
 
    

    #=======================================================================================================================================
    def _get_metrics(total_loss, losses, reg_loss):

        metrics             = {}
        metrics['tot_loss'] = total_loss 
        for data_id, loss in losses.items():
            metrics[data_id + '_loss'] = loss 
        if reg_loss is not None:
            metrics['reg_loss'] = reg_loss  
        return metrics

    #=======================================================================================================================================



    #=======================================================================================================================================
    def _get_metrics_test(total_loss, losses):
        metrics             = {}
        metrics['tot_loss'] = total_loss 
        for data_id, loss in losses.items():
            metrics[data_id + '_loss'] = loss 
        return metrics

    #=======================================================================================================================================   