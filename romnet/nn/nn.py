import numpy as np
import os 
import h5py

import tensorflow_probability                   as tfp

import tensorflow                               as tf
from tensorflow.python.keras                import metrics as metrics_mod
from tensorflow                             import train

from tensorflow.python.keras                import backend
from tensorflow.python.keras.saving         import saving_utils
from tensorflow.python.training.tracking    import util as trackable_utils
from tensorflow.python.keras.utils.io_utils import path_to_string
  
from ..utils                                import hdf5_format
  
from ..training                             import steps as stepss
from ..training                             import losscontainer


# pylint: disable=g-import-not-at-top
try:
  import h5py
except ImportError:
  h5py = None
# pylint: enable=g-import-not-at-top



#=======================================================================================================================================
def load_model_(ModelFld):

    NN = tf.keras.Model.load_model(ModelFld)

    return NN

#=======================================================================================================================================



#=======================================================================================================================================
def load_weights_(ModelFld):

    # ModelFile         = ModelFld + '/MyModel/'
    # NN                = tf.keras.Model.load_model(ModelFile)
    # MCFile            = ModelFld + '/Params/ModelCheckpoint/cp-{epoch:04d}.ckpt'
    # checkpoint_dir    = os.path.dirname(MCFile)
    # latest            = train.latest_checkpoint(checkpoint_dir)

    ModelFld = ModelFld + "/Training/Params/"
    last = max(os.listdir(ModelFld), key=lambda x: int(x.split('.')[0]))
    if last:
        ModelFld = ModelFld + "/" + last
    print('\n[ROMNet - nn.py                     ]:   Loading ML Model Parameters from File: ', ModelFld)

    NN.load_weights(ModelFld)

    return NN

#=======================================================================================================================================



#=======================================================================================================================================
@tf.keras.utils.register_keras_serializable(package='ROMNet', name='NN')
class NN(tf.keras.Model):
    """Base class for all surrogate modules."""

    def __init__(self):
        super(NN, self).__init__()

        self.sructure_name  = 'NN'

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
        base_config = super(NN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



    @classmethod
    def from_config(cls, config):
        return cls(**config)


    #=======================================================================================================================================
    def compile(
        self,
        data,
        optimizer           = 'rmsprop',
        loss                = None,
        metrics             = None,
        loss_weights        = None,
        weighted_metrics    = None,
        run_eagerly         = None,
        steps_per_execution = None,
        **kwargs
    ):

        self.data_type = data.Type
        if self.data_type == 'PDE':

            from_serialized = kwargs.pop('from_serialized', False)

            self._validate_compile(optimizer, metrics, **kwargs)
            self._run_eagerly = run_eagerly

            # Defining optimizer
            self.optimizer = self._get_optimizer(optimizer)

            # Defining loss containers
            self.compiled_loss = {}
            for data_id in self.data_ids:
                _loss                                    = loss[data_id] if loss else None
                _loss_weights                            = loss_weights[data_id] if loss_weights else None
                self.compiled_loss[data_id]              = losscontainer.LossesContainer(_loss, loss_weights=_loss_weights, output_names=self.output_vars)
                self.compiled_loss[data_id]._loss_metric = metrics_mod.Mean(name=data_id + '_loss')
            
            # Defining metrics container
            if metrics is not None:
                print( "[ROMNet - nn.py                     ]   WARNING! Metrics evaluation is not available." )
            self.compiled_metrics = None

            self._configure_steps_per_execution(steps_per_execution or 1)

            # Initializes attrs that are reset each time `compile` is called.
            self._reset_compile_cache()
            self._is_compiled = True
            self.loss         = loss or {}

        else:

            return super(NN, self).compile(
                optimizer           = optimizer,
                loss                = loss,
                metrics             = metrics,
                loss_weights        = loss_weights,
                weighted_metrics    = weighted_metrics,
                run_eagerly         = run_eagerly,
                steps_per_execution = steps_per_execution,
                **kwargs)

    #=======================================================================================================================================



    #=======================================================================================================================================
    @property
    def metrics(self):
        
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                if isinstance(self.compiled_loss, dict):
                    for container in self.compiled_loss.values():
                        metrics += container.metrics
                else:
                    metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics
        return metrics

    #=======================================================================================================================================



    # Steps
    #=======================================================================================================================================
    def train_step(self, data):
        if (self.data_type == 'PDE'):
            return stepss.train_step(self, data)
        else:
            return super(NN, self).train_step(data)

    #=======================================================================================================================================



    #=======================================================================================================================================
    def test_step(self, data):
        if (self.data_type == 'PDE'):
            return stepss.test_step(self, data)
        else:
            return super(NN, self).test_step(data)

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


    