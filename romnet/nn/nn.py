import numpy as np
import os 
import h5py

import tensorflow_probability                 as tfp

import tensorflow                             as tf
from tensorflow.python.keras              import metrics as metrics_mod
from tensorflow                           import train

from ..training                           import steps as stepss
from ..training                           import losscontainer



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

