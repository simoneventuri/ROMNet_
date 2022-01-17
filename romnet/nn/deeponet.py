import numpy        as np
import tensorflow   as tf
import pandas       as pd

from tensorflow.python.keras.layers.merge import _Merge

from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export

from .nn        import NN


class DeepONet(NN):
    """Deep operator network 
    """

    #===================================================================================================================================
    def __init__(self, InputData, xnorm, NN_Transfer_Model):
        super(DeepONet, self).__init__()

        self.Name                 = 'DeepONet'

        self.BranchVars           = InputData.BranchVars
        self.TrunkVars            = InputData.TrunkVars
        self.NVarsx               = len(InputData.BranchVars + InputData.TrunkVars)

        if (xnorm is None):
            xnorm = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=self.BranchVars+self.TrunkVars)

        self.xnormBranch          = xnorm[InputData.BranchVars]
        self.xnormTrunk           = xnorm[InputData.TrunkVars] 
        self.NN_Transfer_Model    = NN_Transfer_Model

        self.OutputVars           = InputData.OutputVars
        self.NVarsy               = len(InputData.OutputVars)

        self.NormalizeInput       = InputData.NormalizeInput
        self.WeightDecay          = InputData.WeightDecay

        self.NTrunks              = len(InputData.TrunkLayers)
        self.NVarsTrunk           = len(InputData.TrunkVars)
        self.TrunkLayers          = InputData.TrunkLayers
        self.TrunkActFun          = InputData.TrunkActFun
        self.TrunkDropOutRate     = InputData.TrunkDropOutRate
        self.TrunkDropOutPredFlg  = InputData.TrunkDropOutPredFlg

        self.NBranches            = len(InputData.BranchLayers)
        self.NVarsBranch          = len(InputData.BranchVars)
        self.BranchLayers         = InputData.BranchLayers
        self.BranchActFun         = InputData.BranchActFun
        self.BranchDropOutRate    = InputData.BranchDropOutRate
        self.BranchDropOutPredFlg = InputData.BranchDropOutPredFlg
        self.BranchSoftmaxFlg     = InputData.BranchSoftmaxFlg
        self.SoftMaxFlg           = False
        self.BranchToTrunk        = InputData.BranchToTrunk

        self.FinalLayerFlg        = InputData.FinalLayerFlg

        self.TransFun             = InputData.TransFun

        if (np.sum(np.array(self.WeightDecay)) > 0.):
            self.RegularizeFlg    = True
        else:   
            self.RegularizeFlg    = False
   
        self.attention_mask       = None

        try:
            self.PathToPODFile    = InputData.PathToPODFile
        except:
            self.PathToPODFile    = None

        try:
            self.TrainTrunkFlg    = InputData.TrainTrunkFlg
        except:
            self.TrainTrunkFlg    = True

        try:
            self.TrainBranchFlg   = InputData.TrainBranchFlg
        except:
            self.TrainBranchFlg   = True


        ### Trunks
        self.TrunkLayersVecs = {}
        for iTrunk in range(self.NTrunks):
            self.TrunkLayersVecs[iTrunk] = self.fnn_block(self.xnormTrunk, 'Trunk', 'Trunk_'+str(iTrunk+1), iTrunk, self.TrunkVars, LayersVec=[])


        self.BranchLayersVecs = {}
        self.FinalLayersVecs  = {}
        self.NDot             = InputData.TrunkLayers[0][-1]
        for iy in range(self.NVarsy):

            ### Branches
            self.BranchLayersVecs[iy] = self.fnn_block(self.xnormBranch, 'Branch', 'Branch_'+InputData.OutputVars[iy], iy, self.BranchVars, LayersVec=[])

            if (self.BranchSoftmaxFlg):

                Layer_ = tf.keras.layers.Dense(units              = InputData.TrunkLayers[0][-1]+1,
                                               activation         = 'linear',
                                               use_bias           = False,
                                               name               = 'Branch_'+InputData.OutputVars[iy]+'_POD')
                Layer_.trainable = False
                self.BranchLayersVecs[iy].append(Layer_)
       
            ### Final Layer
            if (self.FinalLayerFlg):
                self.FinalLayersVecs[iy]  = self.deeponet_final_layer(iy, 'FinalScaling_')

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        inputsBranch, inputsTrunk = tf.split(inputs, num_or_size_splits=[len(self.BranchVars), len(self.TrunkVars)], axis=1)
    

        TrunkVec = []
        for iTrunk in range(self.NTrunks):
            y = inputsTrunk
            
            for f in self.TrunkLayersVecs[iTrunk]:
                if ('dropout' in f.name):
                    y = f(y, training=(training or self.TrunkDropOutPredFlg))
                else:
                    y = f(y, training=training)

            TrunkVec.append(y)


        OutputVec = []        
        for iy in range(self.NVarsy):
            iTrunk = self.BranchToTrunk[iy]
            y      = inputsBranch

            for f in self.BranchLayersVecs[iy]:
                if ('dropout' in f.name):
                    y = f(y, training=(training or self.BranchDropOutPredFlg))
                else:
                    y = f(y, training=training)

            OutputP = Dot_Add(axes=1, n_out=self.NDot)([y, TrunkVec[iTrunk]])
            #OutputP = tf.keras.layers.Dot(axes=1)([y, TrunkVec[iTrunk]])

            if (self.FinalLayerFlg):
                OutputVec.append( self.FinalLayersVecs[iy](OutputP, training=training) )
            else:
                OutputVec.append( OutputP )
                

        if (self.NVarsy > 1):
            OutputConcat = tf.keras.layers.Concatenate(axis=1)(OutputVec)
        else:
            OutputConcat = OutputVec[0]


        if (self.AntiPCA_flg):
            OutputFinal = ROM.AntiPCALayer(A0=self.A_AntiPCA, C0=self.C_AntiPCA, D0=self.D_AntiPCA)(OutputConcat)
        else:
            OutputFinal = OutputConcat


        return OutputFinal

    #===================================================================================================================================



    #===================================================================================================================================
    def get_graph(self):
        input_     = tf.keras.Input(shape=[self.NVarsx,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.layers.Dot_Add')
class Dot_Add(_Merge):
    """Layer that computes a dot product between samples in two tensors.
    E.g. if applied to a list of two tensors `a` and `b` of shape
    `(batch_size, n)`, the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.
    >>> x = np.arange(10).reshape(1, 5, 2)
    >>> print(x)
    [[[0 1]
        [2 3]
        [4 5]
        [6 7]
        [8 9]]]
    >>> y = np.arange(10, 20).reshape(1, 2, 5)
    >>> print(y)
    [[[10 11 12 13 14]
        [15 16 17 18 19]]]
    >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
    <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
    array([[[260, 360],
                    [320, 445]]])>
    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> dotted = tf.keras.layers.Dot(axes=1)([x1, x2])
    >>> dotted.shape
    TensorShape([5, 1])
    """

    def __init__(self, axes, n_out, normalize=False, **kwargs):
        """Initializes a layer that computes the element-wise dot product.
            >>> x = np.arange(10).reshape(1, 5, 2)
            >>> print(x)
            [[[0 1]
                [2 3]
                [4 5]
                [6 7]
                [8 9]]]
            >>> y = np.arange(10, 20).reshape(1, 2, 5)
            >>> print(y)
            [[[10 11 12 13 14]
                [15 16 17 18 19]]]
            >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
            <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
            array([[[260, 360],
                            [320, 445]]])>
        Args:
            axes: Integer or tuple of integers,
                axis or axes along which to take the dot product. If a tuple, should
                be two integers corresponding to the desired axis from the first input
                and the desired axis from the second input, respectively. Note that the
                size of the two selected axes must match.
            normalize: Whether to L2-normalize samples along the
                dot product axis before taking the dot product.
                If set to True, then the output of the dot product
                is the cosine proximity between the two samples.
            **kwargs: Standard layer keyword arguments.
        """
        super(Dot_Add, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError(
                        'Invalid type for argument `axes`: it should be '
                        f'a list or an int. Received: axes={axes}')
            if len(axes) != 2:
                raise ValueError(
                        'Invalid format for argument `axes`: it should contain two '
                        f'elements. Received: axes={axes}')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError(
                        'Invalid format for argument `axes`: list elements should be '
                        f'integers. Received: axes={axes}')
        self.axes              = axes
        self.n_out             = n_out
        self.normalize         = normalize
        self.supports_masking  = True
        self._reshape_required = False


    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], tuple) or len(input_shape) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on a list of 2 inputs. '
                    f'Received: input_shape={input_shape}')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]+2:
            raise ValueError(
                    'Incompatible input shapes: '
                    f'axis values {shape1[axes[0]]} (at axis {axes[0]}) != '
                    f'{shape2[axes[1]]*2} (at axis {axes[1]}). '
                    f'Full input shapes: {shape1}, {shape2}')


    def _merge_function(self, inputs):
        base_layer_utils.no_ragged_support(inputs, self.name)
        if len(inputs) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on exactly 2 inputs. '
                    f'Received: inputs={inputs}')
        x1         = inputs[1]
        x2, x3, x4 = tf.split(inputs[0], num_or_size_splits=[self.n_out, 1, 1], axis=1)
        print(x4)
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % backend.ndim(x1), self.axes % backend.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % backend.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = tf.linalg.l2_normalize(x1, axis=axes[0])
            x2 = tf.linalg.l2_normalize(x2, axis=axes[1])

        output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True)*x4 + x3
        return output


    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on a list of 2 inputs. '
                    f'Received: input_shape={input_shape}')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)


    def compute_mask(self, inputs, mask=None):
        return None


    def get_config(self):
        config = {
                'axes': self.axes,
                'normalize': self.normalize,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#=======================================================================================================================================
