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
        self.SoftMaxFlg        = False

        self.BatchSize         = 0
 
        self.SigmaLike         = 0.

        self.attention_mask    = None
        self.residual          = None

        self.TrainTrunkFlg     = True
        self.TrainBranchFlg    = True
        

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

            #if (iLayer < NLayers-1):
            kW1_ = kW1
            kW2_ = kW2
            kb1_ = kW1
            kb2_ = kW2
            # else:
            #     kb1_ = kW1
            #     kb2_ = kW2
            #     if (BlockName == 'Branch'):
            #         kW1_ = 1.e-7
            #         kW2_ = 0.
            #     else:
            #         kW1_ = kW1
            #         kW2_ = kW2

            if (self.NN_Transfer_Model is not None):
                x0     = self.NN_Transfer_Model.get_layer(layer_name).kernel.numpy()
                b0     = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()
                WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                WRegul = L1L2Regularizer(kW1_, kW2_, x0)
                bRegul = L1L2Regularizer(kb1_, kWb_, b0)
            else:
                if (ActFun[iLayer] == 'relu'):
                    WIni   = 'he_normal'
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

            if (BlockName == 'Trunk') and (not self.TrainTrunkFlg) and (iLayer < NLayers):
                Layer_.trainable = False

            if (BlockName == 'Branch') and (not self.TrainBranchFlg):
                if (not self.BranchSoftmaxFlg):
                    if (iLayer < NLayers):
                        Layer_.trainable = False
                else:
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



        if ( ((BlockName == 'Branch') and (self.BranchSoftmaxFlg)) or (self.SoftMaxFlg) ):

            ### SoftMax Layer 

            LayersVec.append( tf.keras.layers.Softmax() )   



        return LayersVec

    #=======================================================================================================================================


    #=======================================================================================================================================
    def fnn_first_half_block_tfp(self, xnorm, BlockName, NNName, Idx, InputVars, LayersVec=[]):

        if (BlockName != ''):
            VarName = ''
        else:
            VarName = '_' + self.OutputVars[Idx]
            
        if ('POD' in self.OutputVars[Idx]):
            VarName = ''
            NNName  = 'Trunk_'+str(Idx+1)


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
        return LayersVec

    #=======================================================================================================================================



    #=======================================================================================================================================
    def fnn_second_half_block_tfp(self, BlockName, NNName, Idx, Type, LayersVec=[]):
        Idx_ = Idx//2

        if (Type != ''):
            Type = '_'+Type

        # Define the prior weight distribution as Normal of mean=0 and stddev=1.
        # Note that, in this example, the we prior distribution is not trainable,
        # as we fix its parameters.
        def prior(kernel_size, bias_size, pars0, dtype=tf.float64):
            n = kernel_size + bias_size
            prior_model = tf.keras.Sequential(
                [
                    tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.MultivariateNormalDiag(
                            loc=pars0, scale_diag=tf.ones(n, dtype=dtype)
                        )
                    )
                ]
            )
            return prior_model

        # Define variational posterior weight distribution as multivariate Gaussian.
        # Note that the learnable parameters for this distribution are the means,
        # variances, and covariances.
        def posterior(kernel_size, bias_size, dtype=tf.float64):
            n = kernel_size + bias_size
            posterior_model = tf.keras.Sequential(
                [
                    # tfp.layers.VariableLayer(
                    #     tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
                    # ),
                    # tfp.layers.MultivariateNormalTriL(n),
                    tfp.layers.VariableLayer(
                        tfp.layers.IndependentNormal.params_size(n), dtype=dtype
                    ),
                    tfp.layers.IndependentNormal(n),
                ]
            )
            return posterior_model

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)
        bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.BatchSize * 1.0)
    
        if (BlockName != ''):
            VarName = ''
        else:
            VarName = '_' + self.OutputVars[Idx_]
            
        if ('POD' in self.OutputVars[Idx_]):
            VarName = ''
            NNName  = 'Trunk_'+str(Idx_+1)+Type


        ### Hidden Layers

        kW1      = self.WeightDecay[0]
        kW2      = self.WeightDecay[1]
        NNLayers = getattr(self, BlockName+'Layers')[Idx]
        NLayers  = len(NNLayers)
        ActFun   = getattr(self, BlockName+'ActFun')[Idx]

        for iLayer in range(NLayers):
            WeightsName = NNName + VarName + '_HL' + str(iLayer+1) 
            layer_name  = WeightsName 
            if (iLayer == 0):
                NIn = self.NVarsx
            else:
                NIn = NNLayers[iLayer-1]
            NOut = NNLayers[iLayer]

            if (Type == '_') and (iLayer<NLayers-1):
                Layer_ = tfp.layers.DenseLocalReparameterization(units              = np.int32(NNLayers[iLayer]),
                                                                 activation           = ActFun[iLayer],
                                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                                 bias_divergence_fn   = bias_divergence_fn,
                                                                 name                 = layer_name)

                # Layer_ = tfp.layers.DenseFlipout(units              = np.int32(NNLayers[iLayer]),
                #                                activation           = ActFun[iLayer],
                #                                bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                #                                bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                #                                kernel_divergence_fn = kernel_divergence_fn,
                #                                bias_divergence_fn   = bias_divergence_fn,
                #                                name                 = layer_name)
                
                # if (self.PathToLoadFile is not None):
                #     f       = h5py.File(self.PathToLoadFile, 'r')
                #     dset    = f[layer_name+'/'+layer_name+'/']
                #     x0      = dset['kernel:0'][...].flatten()
                #     b0      = dset['bias:0'][...]
                #     f.close()
                # else:
                #     x0      = np.zeros(np.int32(NIn*NOut), dtype=np.float64)
                #     b0      = np.zeros(np.int32(NOut),     dtype=np.float64)
                # Layer_  = DenseVariational_Mod(     units             = NOut,
                #                                     kernel0           = x0,
                #                                     bias0             = b0, 
                #                                     make_prior_fn     = prior,
                #                                     make_posterior_fn = posterior,
                #                                     kl_weight         = 1. / self.BatchSize,
                #                                     activation        = ActFun[iLayer],
                #                                     name              = layer_name)

            else:
                kW1_ = kW1
                kW2_ = kW2
                kb1_ = kW1
                kb2_ = kW2

                if (self.NN_Transfer_Model is not None):
                    x0     = self.NN_Transfer_Model.get_layer(layer_name).kernel.numpy()
                    b0     = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()
                    WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                    bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                    WRegul = L1L2Regularizer(kW1_, kW2_, x0)
                    bRegul = L1L2Regularizer(kb1_, kWb_, b0)
                else:
                    if (ActFun[iLayer] == 'relu'):
                        WIni   = 'he_normal'
                    else:
                        WIni   = 'glorot_normal'
                    bIni   = 'zeros'
                    WRegul = regularizers.l1_l2(l1=kW1_, l2=kW2_)
                    bRegul = regularizers.l1_l2(l1=kb1_, l2=kb2_)

                Layer_ = tf.keras.layers.Dense(units              = NOut,
                                               activation         = ActFun[iLayer],
                                               use_bias           = True,
                                               kernel_initializer = WIni,
                                               bias_initializer   = bIni,
                                               kernel_regularizer = WRegul,
                                               #bias_regularizer   = bRegul,
                                               name               = layer_name)
                
                

            if (BlockName == 'Trunk') and (not self.TrainTrunkFlg) and (iLayer < NLayers):
                Layer_.trainable = False

            if (BlockName == 'Branch') and (not self.TrainBranchFlg):
                if (not self.BranchSoftmaxFlg):
                    if (iLayer < NLayers):
                        Layer_.trainable = False
                else:
                    Layer_.trainable = False

            LayersVec.append(Layer_)

        return LayersVec

    #=======================================================================================================================================



    #=======================================================================================================================================
    def deeponet_final_layer(self, Idx, Name):

        layer_name      = Name + self.OutputVars[Idx]
        if (self.FinalLayerFlg == 'Linear'):
            if (self.NN_Transfer_Model is not None):
                x0     = self.NN_Transfer_Model.get_layer(layer_name).kernel.numpy()
                b0     = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()
                WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            else:
                WIni   = 'he_normal'
                bIni   = 'zeros'
            OutLayer   = tf.keras.layers.Dense(units              = 1,
                                               activation         = 'linear',
                                               use_bias           = True,
                                               kernel_initializer = WIni,
                                               bias_initializer   = bIni,
                                               name               = layer_name)
        elif (self.FinalLayerFlg == 'Shift'):
            b0 = 0
            if (self.NN_Transfer_Model is not None): 
                b0 = self.NN_Transfer_Model.get_layer(layer_name).bias.numpy()[0]
            OutLayer = bias_layer(b0=b0, layer_name=layer_name)

        return OutLayer

    #=======================================================================================================================================


#===================================================================================================================================
class DenseVariational_Mod(tf.keras.layers.Layer):
    """Dense layer with random `kernel` and `bias`.
    This layer uses variational inference to fit a "surrogate" posterior to the
    distribution over both the `kernel` matrix and the `bias` terms which are
    otherwise used in a manner similar to `tf.keras.layers.Dense`.
    This layer fits the "weights posterior" according to the following generative
    process:
    ```none
    [K, b] ~ Prior()
    M = matmul(X, K) + b
    Y ~ Likelihood(M)
    ```
    """

    def __init__(self,
                 units,
                 kernel0,
                 bias0,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        """Creates the `DenseVariational` layer.
        Args:
            units: Positive integer, dimensionality of the output space.
            make_posterior_fn: Python callable taking `tf.size(kernel)`,
                `tf.size(bias)`, `dtype` and returns another callable which takes an
                input and produces a `tfd.Distribution` instance.
            make_prior_fn: Python callable taking `tf.size(kernel)`, `tf.size(bias)`,
                `dtype` and returns another callable which takes an input and produces a
                `tfd.Distribution` instance.
            kl_weight: Amount by which to scale the KL divergence loss between prior
                and posterior.
            kl_use_exact: Python `bool` indicating that the analytical KL divergence
                should be used rather than a Monte Carlo approximation.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseVariational_Mod, self).__init__(activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)

        self.kernel0            = kernel0
        self.bias0              = bias0
        self.pars0              = tf.concat([self.kernel0, self.bias0], 0)

        self._make_posterior_fn = make_posterior_fn
        self._make_prior_fn     = make_prior_fn
        self._kl_divergence_fn  = _make_kl_divergence_penalty(kl_use_exact, weight=kl_weight)

        self.activation         = tf.keras.activations.get(activation)
        self.use_bias           = use_bias
        self.supports_masking   = False
        self.input_spec         = tf.keras.layers.InputSpec(min_ndim=2)


    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))

        input_shape = tf.TensorShape(input_shape)
        last_dim    = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `DenseVariational` should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        self._posterior = self._make_posterior_fn(
                last_dim * self.units,
                self.units if self.use_bias else 0,
                dtype)

        #pars0 = tf.zeros(tf.int32(last_dim * self.units + self.units), dtype=dtype)
        self._prior = self._make_prior_fn(
                last_dim * self.units,
                self.units if self.use_bias else 0,
                self.pars0, dtype)

        self.built = True


    def call(self, inputs):
        dtype  = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)
        prev_units = self.input_spec.axes[-1]
        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
            kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
                tf.shape(kernel)[:-1],
                [prev_units, self.units],
        ], axis=0))
        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return outputs


def _make_kl_divergence_penalty(
        use_exact_kl=False,
        test_points_reduce_axis=(),  # `None` == "all"; () == "none".
        test_points_fn=tf.convert_to_tensor,
        weight=None):
    """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

    if use_exact_kl:
        kl_divergence_fn = kullback_leibler.kl_divergence
    else:
        def kl_divergence_fn(distribution_a, distribution_b):
            z = test_points_fn(distribution_a)
            return tf.reduce_mean(
                    distribution_a.log_prob(z) - distribution_b.log_prob(z),
                    axis=test_points_reduce_axis)

    # Closure over: kl_divergence_fn, weight.
    def _fn(distribution_a, distribution_b):
        """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
        with tf.name_scope('kldivergence_loss'):
            kl = kl_divergence_fn(distribution_a, distribution_b)
            if weight is not None:
                kl = tf.cast(weight, dtype=kl.dtype) * kl
            # Losses appended with the model.add_loss and are expected to be a single
            # scalar, unlike model.loss, which is expected to be the loss per sample.
            # Therefore, we reduce over all dimensions, regardless of the shape.
            # We take the sum because (apparently) Keras will add this to the *post*
            # `reduce_sum` (total) loss.
            # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
            # align, particularly wrt how losses are aggregated (across batch
            # members).
            return tf.reduce_sum(kl, name='batch_total_kl_divergence')

    return _fn
#===================================================================================================================================

