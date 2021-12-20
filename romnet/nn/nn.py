import numpy as np
import os 

import tensorflow                             as tf
from tensorflow.keras                     import regularizers
from tensorflow.keras                     import activations
from tensorflow.keras                     import initializers
from tensorflow.python.ops                import array_ops

# from tensorflow.python.keras.engine       import compile_utils
from tensorflow.python.keras.engine       import base_layer
from tensorflow.python.keras              import metrics as metrics_mod

import tensorflow_probability                 as tfp

from tensorflow                           import train

import h5py

from ..training                           import steps as stepss
from ..training                           import L1L2Regularizer
from ..training                           import losscontainer
from .normalization                       import CustomNormalization


#=======================================================================================================================================
class TransLayer(tf.keras.layers.Layer):

    def __init__(self, f, NVars, indxs, name='TransLayer'):
        super(TransLayer, self).__init__(name=name, trainable=False)
        self.f           = f
        self.NVars       = NVars
        self.indxs       = indxs

    def call(self, inputs):

        inputs_unpack = tf.split(inputs, self.NVars, axis=1)
        
        if (self.f == 'log10'):
            for indx in self.indxs:
                inputs_unpack[indx] = tf.experimental.numpy.log10(inputs_unpack[indx])
        elif (self.f == 'log'):
            for indx in self.indxs:
                inputs_unpack[indx] = tf.math.log(inputs_unpack[indx])
        
        return tf.concat(inputs_unpack, axis=1)
        
#=======================================================================================================================================



#=======================================================================================================================================
class bias_layer(tf.keras.layers.Layer):

    def __init__(self, b0, layer_name):
        super(bias_layer, self).__init__(name=layer_name)
        self.b0   = b0

    def build(self, input_shape):
        bIni      = tf.keras.initializers.constant(value=self.b0)
        self.bias = self.add_weight('bias',
                                    shape       = input_shape[1:],
                                    initializer = bIni,
                                    trainable   = True)

    def call(self, x):
        return x + self.bias

#=======================================================================================================================================



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

    print('\n[ROMNet]:   Loading ML Model Parameters from File: ', ModelFld)

    NN.load_weights(ModelFld)

    return NN

#=======================================================================================================================================



#=======================================================================================================================================
@tf.keras.utils.register_keras_serializable(package='ROMNet', name='NN')
class NN(tf.keras.Model):
    """Base class for all surrogate modules."""

    def __init__(self):
        super(NN, self).__init__()

        self.NormalizeInput    = False
        self.WeightDecay       = [0., 0.]
        self.OutputVars        = []
        self.NVarsy            = 0

        self.Layers            = []
        self.ActFun            = []
        self.DropOutRate       = 0.
        self.DropOutPredFlg    = False
        self.BranchSoftmaxFlg  = False

        self.BatchSize         = 0
 
        self.SigmaLike         = 0.

        self.attention_mask    = None
        self.residual          = None

        self.TrainTrunkFlg     = False
        

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
                self.compiled_loss[data_id]              = losscontainer.LossesContainer(_loss, loss_weights=_loss_weights, output_names=self.OutputVars)
                self.compiled_loss[data_id]._loss_metric = metrics_mod.Mean(name=data_id + '_loss')
            print("[ROMNet]   self.compiled_loss = ", self.compiled_loss)
            
            # Defining metrics container
            if metrics is not None:
                print( "[ROMNet]   WARNING! Metrics evaluation is not available." )
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
    def fnn_block(self, xnorm, BlockName, NNName, Idx, InputVars, LayersVec=[]):

        if (BlockName != ''):
            VarName = ''
        else:
            VarName = '_' + self.OutputVars[Idx]
            
        if ('POD' in self.OutputVars[Idx]):
            VarName = ''
            NNName  = 'Trunk_1'


        ### Transform Layer

        if (self.TransFun):
            for ifun, fun in enumerate(self.TransFun):
                vars_list = self.TransFun[fun]

                indxs = []
                for ivar, var in enumerate(InputVars):
                    if var in vars_list:
                        indxs.append(ivar)

                if (len(indxs) > 0):
                    layer_name = NNName + VarName + '_Transformation_' + fun
                    LayersVec.append( TransLayer(fun, len(InputVars), indxs, name=layer_name) )
            

        ### Normalizer Layer

        if (self.NormalizeInput):
            # if (self.NN_Transfer_Model is not None): 
            #     Mean        = self.NN_Transfer_Model.get_layer('normalization').mean.numpy()
            #     Variance    = self.NN_Transfer_Model.get_layer('normalization').variance.numpy()
            #     normalizer  = tf.keras.layers.Normalization(mean=Mean, variance=Variance)
            # else:
            #     normalizer  = tf.keras.layers.Normalization()
            #     normalizer.adapt(np.array(xnorm))
            # LayersVec.append( normalizer )
            layer_name = NNName + VarName + '_Normalization'
            if ( (BlockName == 'Trunk') and (self.PathToPODFile) ):
                with h5py.File(self.PathToPODFile, "r") as f:
                    Key_       = 'NN_POD_1_Normalization'
                    Mean       = np.array(f[Key_+'/mean:0'][:])
                    Variance   = np.array(f[Key_+'/variance:0'][:])[...,np.newaxis]
                    MinVals    = np.array(f[Key_+'/min_vals:0'][:])[...,np.newaxis]
                    MaxVals    = np.array(f[Key_+'/max_vals:0'][:])[...,np.newaxis]
                    normalizer = CustomNormalization(mean=Mean, variance=Variance, min_vals=MinVals, max_vals=MaxVals, name=layer_name)
            elif (self.NN_Transfer_Model is not None): 
                Mean        = self.NN_Transfer_Model.get_layer(layer_name).mean.numpy()
                Variance    = self.NN_Transfer_Model.get_layer(layer_name).variance.numpy()
                MinVals     = self.NN_Transfer_Model.get_layer(layer_name).min_vals.numpy()
                MaxVals     = self.NN_Transfer_Model.get_layer(layer_name).max_vals.numpy()
                normalizer  = CustomNormalization(mean=Mean, variance=Variance, min_vals=MinVals, max_vals=MaxVals, name=layer_name)
            else:
                normalizer  = CustomNormalization(name=layer_name)
                normalizer.adapt(np.array(xnorm))
            LayersVec.append( normalizer )


        ### Hidden Layers

        kW1      = self.WeightDecay[0]
        kW2      = self.WeightDecay[1]
        NNLayers = getattr(self, BlockName+'Layers')[Idx]
        NLayers  = len(NNLayers)
        ActFun   = getattr(self, BlockName+'ActFun')[Idx]

        for iLayer in range(NLayers):
            WeightsName = NNName + VarName + '_HL' + str(iLayer+1) 
            layer_name   = WeightsName 

            if (iLayer < NLayers-1):
                kW1_ = kW1
                kW2_ = kW2
                kb1_ = kW1
                kb2_ = kW2
            else:
                kb1_ = 0.
                kb2_ = 0.
                if (BlockName == 'Branch'):
                    kW1_ = 1.e-7
                    kW2_ = 0.
                else:
                    kW1_ = kW1
                    kW2_ = kW2

            if (self.NN_Transfer_Model is not None):
                x0     = self.NN_Transfer_Model.get_layer(layer_name).kernel.numpy()
                b0     = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()
                WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                WRegul = L1L2Regularizer(kW1_, kW2_, x0)
                bRegul = L1L2Regularizer(kb1_, kWb_, b0)
            else:
                WIni   = 'glorot_normal'
                bIni   = 'zeros'
                WRegul = regularizers.l1_l2(l1=kW1_, l2=kW2_)
                bRegul = regularizers.l1_l2(l1=kb1_, l2=kb2_)

            Layer_ = tf.keras.layers.Dense(units              = NNLayers[iLayer],
                                           activation         = ActFun[iLayer],
                                           use_bias           = True,
                                           kernel_initializer = WIni,
                                           bias_initializer   = bIni,
                                           kernel_regularizer = WRegul,
                                           #bias_regularizer   = bRegul,
                                           name               = layer_name)
            
            # if ( (BlockName == 'Trunk') and (self.PathToPODFile) ):
            #     with h5py.File(self.PathToPODFile, "r") as f:
            #         Key_   = 'NN_POD_1_HL' + str(iLayer+1) 
            #         x0     = np.array(f[Key_+'/'+Key_+'/kernel:0'][:])
            #         if (iLayer == 0):
            #             x0 = x0[0,:]
            #         b0     = np.array(f[Key_+'/'+Key_+'/bias:0'][:])[...,np.newaxis]
            #         print(Key_, ': x0.shape = ', x0.shape, '; b0.shape = ', b0.shape)
            #         Layer_.set_weights([x0, b0])

            if (BlockName == 'Trunk'):
                if (not self.TrainTrunkFlg):
                    Layer_.trainable = False
            LayersVec.append(Layer_)



            if (iLayer < NLayers-1):# or (BlockName == 'Branch'):
                DropOutRate            = getattr(self, BlockName+'DropOutRate')
                DropOutPredFlg         = getattr(self, BlockName+'DropOutPredFlg')
                DropOutLayer           = tf.keras.layers.Dropout(DropOutRate, input_shape=(NNLayers[iLayer],))
                DropOutLayer.trainable = DropOutPredFlg
                LayersVec.append( DropOutLayer )



        # #if (BlockName in ['','Trunk']):
        # if (BlockName in ['']):

        #     ### Final Layer

        #     NNNs        = len(self.Layers)
        #     NOutputsNN  = len(self.OutputVars)
        #     if (NNNs > 1):
        #         NOutputsNN = 1
     
        #     layer_name      = 'FinalScaling_' + str(Idx+1)
        #     if (self.NN_Transfer_Model is not None):
        #         if (self.NN_Transfer_POD_Flg):
        #             layer_name_ = 'FinalScaling_' + str(Idx+1)
        #         else:
        #             layer_name_ = layer_name
        #         x0     = self.NN_Transfer_Model.get_layer(layer_name_).kernel.numpy()
        #         b0     = self.NN_Transfer_Model.get_layer(layer_name_).bias.numpy()
        #         WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
        #         bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        #         WRegul = L1L2Regularizer(kW1, kW2, x0)
        #     else:
        #         WIni   = 'glorot_normal'
        #         bIni   = 'zeros'
        #     OutLayer   = tf.keras.layers.Dense(units              = NOutputsNN,
        #                                        activation         = 'linear',
        #                                        use_bias           = True,
        #                                        kernel_initializer = WIni,
        #                                        kernel_regularizer = WRegul,
        #                                        bias_initializer   = bIni,
        #                                        name               = layer_name)
            
        #     LayersVec.append( OutLayer )



        if ((BlockName == 'Branch') and (self.BranchSoftmaxFlg)):

            ### SoftMax Layer 

            LayersVec.append( tf.keras.layers.Softmax() )   



        return LayersVec

    #=======================================================================================================================================



    #=======================================================================================================================================
    def fnn_block_tfp(self, xnorm, BlockName, NNName, Idx):

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)
        bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)
    
        if (BlockName == ''):
            VarName = '_' + self.OutputVars[Idx]
        else:
            VarName = ''

        LayersVec = []


        ### Normalizer Layer
        
        if (self.NormalizeInput):
        #     if (self.NN_Transfer_Model is not None): 
        #         Mean        = self.NN_Transfer_Model.get_layer('normalization').mean.numpy()
        #         Variance    = self.NN_Transfer_Model.get_layer('normalization').variance.numpy()
        #         normalizer  = tf.keras.layers.Normalization(mean=Mean, variance=Variance)
        #     else:
        #         normalizer  = tf.keras.layers.Normalization()
        #         normalizer.adapt(np.array(xnorm))
        #     LayersVec.append( normalizer )
            layer_name = NNName + VarName + '_Normalization'
            if (self.NN_Transfer_Model is not None): 
                Mean        = self.NN_Transfer_Model.get_layer(layer_name).mean.numpy()
                Variance    = self.NN_Transfer_Model.get_layer(layer_name).variance.numpy()
                MinVals     = self.NN_Transfer_Model.get_layer(layer_name).min_vals.numpy()
                MaxVals     = self.NN_Transfer_Model.get_layer(layer_name).max_vals.numpy()
                normalizer  = CustomNormalization(mean=Mean, variance=Variance, min_vals=MinVals, max_vals=MaxVals, name=layer_name)
            else:
                normalizer  = CustomNormalization(name=layer_name)
                normalizer.adapt(np.array(xnorm))
            LayersVec.append( normalizer )


        ### Hidden Layers

        NNLayers = getattr(self, BlockName+'Layers')[Idx]
        NLayers  = len(NNLayers)
        ActFun   = getattr(self, BlockName+'ActFun')[Idx]

        for iLayer in range(NLayers):
            WeightsName = NNName + VarName + '_HL' + str(iLayer+1) 
            layer_name   = WeightsName 

            LayersVec.append(tfp.layers.DenseFlipout(units                = np.int32(NNLayers[iLayer]),
                                                     activation           = ActFun[iLayer],
                                                     bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                     bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                     kernel_divergence_fn = kernel_divergence_fn,
                                                     bias_divergence_fn   = bias_divergence_fn,
                                                     name                 = layer_name))

            # if (iLayer < NLayers-1):
            #     DropOutRate            = getattr(self, BlockName+'DropOutRate')
            #     DropOutPredFlg         = getattr(self, BlockName+'DropOutPredFlg')
            #     DropOutLayer           = tf.keras.layers.Dropout(DropOutRate, input_shape=(NNLayers[iLayer],))
            #     DropOutLayer.trainable = DropOutPredFlg
            #     LayersVec.append( DropOutLayer )


        # if (BlockName == ''):

        #     ### Final Layer

        #     NNNs        = len(self.Layers)
        #     NOutputsNN  = self.NVarsy
        #     if (self.SigmaLike is not None):
        #         CalibrateSigmaLFlg = False
        #         NOutputsNN         = np.int32(self.NVarsy)
        #     else:
        #         CalibrateSigmaLFlg = True
        #         NOutputsNN         = np.int32(self.NVarsy * 2)

        #     layer_name      = 'FinalScaling_' + str(Idx+1)
        #     OutLayer       = tfp.layers.DenseFlipout(units                = np.int32(NOutputsNN),
        #                                              activation           = 'linear',
        #                                              bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
        #                                              bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
        #                                              kernel_divergence_fn = kernel_divergence_fn,
        #                                              bias_divergence_fn   = bias_divergence_fn,
        #                                              name                 = layer_name)

        #     LayersVec.append( OutLayer )


        if ((BlockName == 'Branch') and (self.BranchSoftmaxFlg)):

            ### SoftMax Layer 
            
            LayersVec.append(tf.keras.layers.Softmax())        


        return hiddenVec[-1]

    #=======================================================================================================================================



    #=======================================================================================================================================
    def deeponet_final_layer(self, Idx):

        layer_name      = 'FinalScaling_' + self.OutputVars[Idx]
        if (self.FinalLayerFlg):
            if (self.NN_Transfer_Model is not None):
                x0     = self.NN_Transfer_Model.get_layer(layer_name).kernel.numpy()
                b0     = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()
                WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            else:
                WIni   = 'glorot_normal'
                bIni   = 'zeros'
            OutLayer   = tf.keras.layers.Dense(units              = 1,
                                               activation         = 'linear',
                                               use_bias           = True,
                                               kernel_initializer = WIni,
                                               bias_initializer   = bIni,
                                               name               = layer_name)
        else:
            b0 = 0
            if (self.NN_Transfer_Model is not None): 
                b0 = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()[0]
            OutLayer = bias_layer(b0=b0, layer_name=layer_name)

        return OutLayer

    #=======================================================================================================================================



    #=======================================================================================================================================
    def deeponet_final_layer_tfp(self, Idx, OutputVars):

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)
        bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)

        ### Final Layer
        layer_name      = 'FinalScaling_' + OutputVars[Idx]
        # if (self.FinalLayerFlg):
        OutLayer       = tfp.layers.DenseFlipout(units                = 1,
                                                 activation           = 'linear',
                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                 bias_divergence_fn   = bias_divergence_fn,
                                                 name                 = layer_name)


        # else:
        #     b0 = 0
        #     if (self.NN_Transfer_Model is not None): 
        #         b0 = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()[0]
        #     output_net = bias_layer(b0=b0, layer_name=layer_name)(output_P)

        return OutLayer

    #=======================================================================================================================================








#=======================================================================================================================================