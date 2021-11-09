import numpy as np

from tensorflow.keras                     import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras                     import regularizers
from tensorflow.keras                     import activations
from tensorflow.keras                     import initializers
from tensorflow.python.ops                import array_ops

import tensorflow_probability                 as tfp

from ..regularizer import L1L2Regularizer




#=======================================================================================================================================

def fnn_block(InputData, Input, xTrain, BlockName, NNName, Idx, NN_Transfer_Model):


    ### Normalizer Layer
    
    if (InputData.NormalizeInput):
        if (InputData.TransferFlg): 
            Mean        = NN_Transfer_Model.get_layer('normalization').mean.numpy()
            Variance    = NN_Transfer_Model.get_layer('normalization').variance.numpy()
            normalizer  = preprocessing.Normalization(mean=Mean, variance=Variance)
        else:
            normalizer  = preprocessing.Normalization()
            normalizer.adapt(np.array(xTrain))
        Input_          = normalizer(Input)
    else:
        Input_          = Input



    ### Hidden Layers

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = getattr(InputData, BlockName+'Layers')[Idx]
    NLayers  = len(NNLayers)
    ActFun   = getattr(InputData, BlockName+'ActFun')[Idx]

    if (BlockName == ''):
        VarName = '_' + InputData.OutputVars[Idx]
    else:
        VarName = ''

    hiddenVec = [Input_]
    for iLayer in range(NLayers):
        WeightsName = NNName + VarName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        if (InputData.TransferFlg):
            x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
            b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
            WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
            bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            WRegul = L1L2Regularizer(kW1, kW2, x0)
            bRegul = L1L2Regularizer(kW1, kW2, b0)
        else:
            WIni   = 'glorot_normal'
            bIni   = 'zeros'
            WRegul = regularizers.l1_l2(l1=kW1, l2=kW2)
            bRegul = regularizers.l1_l2(l1=kW1, l2=kW2)

        hiddenVec.append(layers.Dense(units              = NNLayers[iLayer],
                                      activation         = ActFun[iLayer],
                                      use_bias           = True,
                                      kernel_initializer = WIni,
                                      bias_initializer   = bIni,
                                      kernel_regularizer = WRegul,
                                      bias_regularizer   = bRegul,
                                      name               = LayerName)(hiddenVec[-1]))
        
        if (iLayer < NLayers-1):
            DropOutRate    = getattr(InputData, BlockName+'DropOutRate')
            DropOutPredFlg = getattr(InputData, BlockName+'DropOutPredFlg')
            hiddenVec.append( layers.Dropout(DropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=DropOutPredFlg) )



    if (BlockName == ''):

        ### Final Layer

        NNNs        = len(InputData.Layers)
        NOutputsNN  = len(InputData.OutputVars)
        if (NNNs > 1):
            NOutputsNN = 1
 
        LayerName      = 'FinalScaling_' + str(Idx+1)
        if (InputData.TransferFlg):
            x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
            b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
            WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
            bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            WIni   = 'glorot_normal'
            bIni   = 'zeros'
        output_net = layers.Dense(units              = NOutputsNN,
                                  activation         = 'linear',
                                  use_bias           = True,
                                  kernel_initializer = WIni,
                                  bias_initializer   = bIni,
                                  name               = LayerName)(hiddenVec[-1]) 
        
        hiddenVec.append(output_net)


    elif ((BlockName == 'Branch') and (InputData.BranchSoftmaxFlg)):

        ### SoftMax Layer 

        hiddenVec.append(layers.Softmax()(hiddenVec[-1]))   



    return hiddenVec[-1]

#=======================================================================================================================================



#=======================================================================================================================================

def fnn_block_tfp(InputData, Input, xTrain, BlockName, NNName, Idx, NN_Transfer_Model):

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
    bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
    


    ### Normalizer Layer
    
    if (InputData.NormalizeInput):
        if (InputData.TransferFlg): 
            Mean        = NN_Transfer_Model.get_layer('normalization').mean.numpy()
            Variance    = NN_Transfer_Model.get_layer('normalization').variance.numpy()
            normalizer  = preprocessing.Normalization(mean=Mean, variance=Variance)
        else:
            normalizer  = preprocessing.Normalization()
            normalizer.adapt(np.array(xTrain))
        Input_          = normalizer(Input)
    else:
        Input_          = Input



    ### Hidden Layers

    NNLayers = getattr(InputData, BlockName+'Layers')[Idx]
    NLayers  = len(NNLayers)
    ActFun   = getattr(InputData, BlockName+'ActFun')[Idx]

    if (BlockName == ''):
        VarName = '_' + InputData.OutputVars[Idx]
    else:
        VarName = ''

    hiddenVec = [Input_]
    for iLayer in range(NLayers):
        WeightsName = NNName + VarName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        hiddenVec.append(tfp.layers.DenseFlipout(units                = np.int32(NNLayers[iLayer]),
                                                 activation           = ActFun[iLayer],
                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                 bias_divergence_fn   = bias_divergence_fn,
                                                 name                 = LayerName)(hiddenVec[-1]))

        # if (iLayer < NLayers-1):
        #     DropOutRate    = getattr(InputData, BlockName+'DropOutRate')
        #     DropOutPredFlg = getattr(InputData, BlockName+'DropOutPredFlg')
        #     hiddenVec.append( layers.Dropout(DropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=DropOutPredFlg) )     



    if (BlockName == ''):

        ### Final Layer

        NNNs        = len(InputData.Layers)
        NOutputsNN  = self.NVarsy
        if (InputData.SigmaLike is not None):
            CalibrateSigmaLFlg = False
            NOutputsNN         = np.int32(self.NVarsy)
        else:
            CalibrateSigmaLFlg = True
            NOutputsNN         = np.int32(self.NVarsy * 2)

        LayerName      = 'FinalScaling_' + str(Idx+1)
        output_net     = tfp.layers.DenseFlipout(units                = np.int32(NOutputsNN),
                                                 activation           = 'linear',
                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                 bias_divergence_fn   = bias_divergence_fn,
                                                 name                 = LayerName)(hiddenVec[-1])

        hiddenVec.append(output_net)


    elif ((BlockName == 'Branch') and (InputData.BranchSoftmaxFlg)):

        ### SoftMax Layer 
        
        hiddenVec.append(layers.Softmax()(hiddenVec[-1]))        


    return hiddenVec[-1]

#=======================================================================================================================================



#=======================================================================================================================================

class bias_layer(layers.Layer):

    def __init__(self, b0, LayerName):
        super(BiasLayer, self).__init__(name=LayerName)
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

def deeponet_final_layer(InputData, output_P, Idx, NN_Transfer_Model):

    LayerName      = 'FinalScaling_' + InputData.OutputVars[Idx]
    if (InputData.FinalLayerFlg):
        if (InputData.TransferFlg):
            x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
            b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
            WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
            bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            WIni   = 'glorot_normal'
            bIni   = 'zeros'
        output_net = layers.Dense(units              = 1,
                                  activation         = 'linear',
                                  use_bias           = True,
                                  kernel_initializer = WIni,
                                  bias_initializer   = bIni,
                                  name               = LayerName)(output_P) 
    else:
        b0 = 0
        if (InputData.TransferFlg): 
            b0 = NN_Transfer_Model.get_layer(LayerName).bias.numpy()[0]
        output_net = bias_layer(b0=b0, LayerName=LayerName)(output_P)

    return output_net

#=======================================================================================================================================



#=======================================================================================================================================

def deeponet_final_layer_tfp(InputData, output_P, Idx, OutputVars, NN_Transfer_Model):

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
    bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)

    ### Final Layer
    LayerName      = 'FinalScaling_' + OutputVars[Idx]
    # if (InputData.FinalLayerFlg):
    output_net     = tfp.layers.DenseFlipout(units                = 1,
                                             activation           = 'linear',
                                             bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                             bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                             kernel_divergence_fn = kernel_divergence_fn,
                                             bias_divergence_fn   = bias_divergence_fn,
                                             name                 = LayerName)(output_P)


    # else:
    #     b0 = 0
    #     if (InputData.TransferFlg): 
    #         b0 = NN_Transfer_Model.get_layer(LayerName).bias.numpy()[0]
    #     output_net = bias_layer(b0=b0, LayerName=LayerName)(output_P)

    return output_net

#=======================================================================================================================================


