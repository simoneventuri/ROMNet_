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
        self.NDotTrunk            = [InputData.TrunkLayers[i][-1] for i in range(self.NTrunks)]

        try:
            self.TrunkULayers       = InputData.TrunkULayers                          
            self.TrunkUActFun       = InputData.TrunkUActFun                          
            self.TrunkVLayers       = InputData.TrunkVLayers                          
            self.TrunkVActFun       = InputData.TrunkVActFun 
        except: 
            self.TrunkULayers       = None                
            self.TrunkUActFun       = None                
            self.TrunkVLayers       = None                
            self.TrunkVActFun       = None  

        self.NBranches            = len(InputData.BranchLayers)
        self.NVarsBranch          = len(InputData.BranchVars)
        self.BranchLayers         = InputData.BranchLayers
        self.BranchActFun         = InputData.BranchActFun
        self.BranchDropOutRate    = InputData.BranchDropOutRate
        self.BranchDropOutPredFlg = InputData.BranchDropOutPredFlg
        self.BranchSoftmaxFlg     = InputData.BranchSoftmaxFlg
        self.SoftMaxFlg           = False
        self.BranchToTrunk        = InputData.BranchToTrunk
        self.NDotBranch           = [InputData.BranchLayers[i][-1] for i in range(self.NBranches)]

        try:
            self.BranchULayers       = InputData.BranchULayers                          
            self.BranchUActFun       = InputData.BranchUActFun                          
            self.BranchVLayers       = InputData.BranchVLayers                          
            self.BranchVActFun       = InputData.BranchVActFun 
        except: 
            self.BranchULayers       = None                
            self.BranchUActFun       = None                
            self.BranchVLayers       = None                
            self.BranchVActFun       = None   

        try:
            self.tShiftLayers         = InputData.tShiftLayers
            self.tShiftActFun         = InputData.tShiftActFun
            self.tShiftDropOutRate    = InputData.tShiftDropOutRate
            self.tShiftDropOutPredFlg = InputData.tShiftDropOutPredFlg
        except:
            self.tShiftLayers         = None
            self.tShiftActFun         = None
            self.tShiftDropOutRate    = None
            self.tShiftDropOutPredFlg = None

        try:
            self.tShiftULayers       = InputData.tShiftULayers                          
            self.tShiftUActFun       = InputData.tShiftUActFun                          
            self.tShiftVLayers       = InputData.tShiftVLayers                          
            self.tShiftVActFun       = InputData.tShiftVActFun 
        except: 
            self.tShiftULayers       = None                
            self.tShiftUActFun       = None                
            self.tShiftVLayers       = None                
            self.tShiftVActFun       = None  


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

        self.ShiftTrunkFlg     = (not self.tShiftLayers is None)
        self.NormalizeTrunkFlg = self.NormalizeInput and (not self.ShiftTrunkFlg)



        ### Trunks
        self.TrunkLayersVecs  = {}
        self.TrunkLayers_U    = {}
        self.TrunkLayers_V    = {}
        self.tShiftLayersVecs = {}
        self.tShiftLayers_U   = {}
        self.tShiftLayers_V   = {}
        for iTrunk in range(self.NTrunks):
            self.TrunkLayersVecs[iTrunk], self.TrunkLayers_U[iTrunk], self.TrunkLayers_V[iTrunk]    = self.fnn_block(self.xnormTrunk, 'Trunk', 'Trunk_'+str(iTrunk+1), iTrunk, self.TrunkVars, self.NormalizeTrunkFlg, LayersVec=[])

        if (self.ShiftTrunkFlg):
            self.tShiftLayersVecs[0], self.tShiftLayers_U[0], self.tShiftLayers_V[0] = self.fnn_block(self.xnormBranch, 'tShift', 'tShift_'+str(0+1), 0, self.BranchVars, self.NormalizeInput, LayersVec=[])


        self.BranchLayersVecs = {}
        self.BranchLayers_U   = {}
        self.BranchLayers_V   = {}
        self.FinalLayersVecs  = {}
        self.NDot             = InputData.TrunkLayers[0][-1]
        for iy in range(self.NVarsy):

            ### Branches
            self.BranchLayersVecs[iy], self.BranchLayers_U[iy], self.BranchLayers_V[iy] = self.fnn_block(self.xnormBranch, 'Branch', 'Branch_'+InputData.OutputVars[iy], iy, self.BranchVars, self.NormalizeInput, LayersVec=[])


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



        ### Trunks
        self.TrunkLayersVecs_bis  = {}
        self.TrunkLayers_U_bis    = {}
        self.TrunkLayers_V_bis    = {}
        self.tShiftLayersVecs_bis = {}
        self.tShiftLayers_U_bis   = {}
        self.tShiftLayers_V_bis   = {}
        for iTrunk in range(self.NTrunks_bis):
            self.TrunkLayersVecs_bis[iTrunk], self.TrunkLayers_U_bis[iTrunk], self.TrunkLayers_V_bis[iTrunk] = self.fnn_block(self.xnormTrunk, 'BisTrunk', 'BisTrunk_'+str(iTrunk+1), iTrunk, self.TrunkVars, self.NormalizeTrunkFlg, LayersVec=[])

        if (self.ShiftTrunkFlg):
            self.tShiftLayersVecs_bis[0], self.tShiftLayers_U_bis[0], self.tShiftLayers_V_bis[0] = self.fnn_block(self.xnormBranch, 'BistShift', 'BistShift_'+str(0+1), 0, self.BranchVars, self.NormalizeInput, LayersVec=[])


        self.BranchLayersVecs_bis = {}
        self.BranchLayers_U_bis   = {}
        self.BranchLayers_V_bis   = {}
        self.FinalLayersVecs_bis  = {}
        for iy in range(self.NVarsy):

            ### Branches
            self.BranchLayersVecs_bis[iy], self.BranchLayers_U_bis[iy], self.BranchLayers_V_bis[iy] = self.fnn_block(self.xnormBranch, 'BisBranch', 'BisBranch_'+InputData.OutputVars[iy], iy, self.BranchVars, self.NormalizeInput, LayersVec=[])


            if (self.BranchSoftmaxFlg):

                Layer_ = tf.keras.layers.Dense(units              = InputData.TrunkLayers[0][-1]+1,
                                               activation         = 'linear',
                                               use_bias           = False,
                                               name               = 'BisBranch_'+InputData.OutputVars[iy]+'_POD')
                Layer_.trainable = False
                self.BranchLayersVecs_bis[iy].append(Layer_)
       
            ### Final Layer
            if (self.FinalLayerFlg):
                self.FinalLayersVecs_bis[iy]  = self.deeponet_final_layer(iy, 'BisFinalScaling_')

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        inputsBranch, inputsTrunk = tf.split(inputs, num_or_size_splits=[len(self.BranchVars), len(self.TrunkVars)], axis=1)
    

        if (self.ShiftTrunkFlg):
            NLayers_tShift = len(self.tShiftLayersVecs[0])
            
            y = inputsBranch
            if (self.NormalizeInput):
                iStart = 1
                y      = self.BranchLayersVecs[0][0](y, training=training)
            else:
                iStart = 0 

            if (self.tShiftLayers_U[0]):
                y_U = self.tShiftLayers_U[0](y, training=training)
                y_V = self.tShiftLayers_V[0](y, training=training)

            for iLayer, f in enumerate(self.tShiftLayersVecs[0][iStart::]):
                if ('dropout' in f.name):
                    y = f(y, training=(training or self.tShiftDropOutPredFlg))
                else:
                    y = f(y, training=training)
                if ( (self.tShiftLayers_U[0]) and (iLayer < NLayers_tShift-(1+iStart)) and (not 'dropout' in f.name) ):
                    yo     = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
                    ya     = tf.keras.layers.multiply([yo, y_U])
                    yb     = tf.keras.layers.multiply([   y, y_V])
                    y      = tf.keras.layers.add([ya, yb])
            tShift = tf.split(y, num_or_size_splits=[1]*self.NVarsy, axis=1)
            print('tShift = ', tShift)


        TrunkVec = []
        for iTrunk in range(self.NTrunks):
            NLayers_Trunk  = len(self.TrunkLayersVecs[iTrunk])
            
            # if (self.ShiftTrunkFlg):
            #     NLayers_tShift = len(self.tShiftLayersVecs[iTrunk])
                
            #     y = inputsBranch
            #     if (self.NormalizeInput):
            #         iStart = 1
            #         y      = self.BranchLayersVecs[0][0](y, training=training)
            #     else:
            #         iStart = 0 

            #     if (self.tShiftLayers_U[iTrunk]):
            #         y_U = self.tShiftLayers_U[iTrunk](y, training=training)
            #         y_V = self.tShiftLayers_V[iTrunk](y, training=training)

            #     for iLayer, f in enumerate(self.tShiftLayersVecs[iTrunk][iStart::]):
            #         if ('dropout' in f.name):
            #             y = f(y, training=(training or self.tShiftDropOutPredFlg))
            #         else:
            #             y = f(y, training=training)
            #         if ( (self.tShiftLayers_U[iTrunk]) and (iLayer < NLayers_tShift-(1+iStart)) and (not 'dropout' in f.name) ):
            #             yo     = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
            #             ya     = tf.keras.layers.multiply([yo, y_U])
            #             yb     = tf.keras.layers.multiply([   y, y_V])
            #             y      = tf.keras.layers.add([ya, yb])
            #     tShift = y


            y = inputsTrunk
            if (self.NormalizeTrunkFlg):
                iStart = 1
                y      = self.TrunkLayersVecs[iTrunk][0](y, training=training)
            else:
                iStart = 0 

            if (self.ShiftTrunkFlg):
                iStart = 1
                y      = self.TrunkLayersVecs[iTrunk][0](y, training=training)    
                y      = tf.keras.layers.subtract([y, tShift[iTrunk]])

            if (self.TrunkLayers_U[iTrunk]):
                y_U = self.TrunkLayers_U[iTrunk](y, training=training)
                y_V = self.TrunkLayers_V[iTrunk](y, training=training)

            for iLayer, f in enumerate(self.TrunkLayersVecs[iTrunk][iStart::]):
                if ('dropout' in f.name):
                    y = f(y, training=(training or self.TrunkDropOutPredFlg))
                else:
                    y = f(y, training=training)
                if ( (self.TrunkLayers_U[iTrunk]) and (iLayer < NLayers_Trunk-(1+iStart)) and (not 'dropout' in f.name) ):
                    yo = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
                    ya = tf.keras.layers.multiply([yo, y_U])
                    yb = tf.keras.layers.multiply([   y, y_V])
                    y  = tf.keras.layers.add([ya, yb])

            TrunkVec.append(y)


        OutputVec = []        
        for iy in range(self.NVarsy):
            iTrunk  = self.BranchToTrunk[iy]
            NLayers = len(self.BranchLayersVecs[iy])

            y = inputsBranch
            if (self.NormalizeInput):
                iStart = 1
                y      = self.BranchLayersVecs[iy][0](y, training=training)
            else:
                iStart = 0 

            if (self.BranchLayers_U[iy]):
                y_U = self.BranchLayers_U[iy](y, training=training)
                y_V = self.BranchLayers_V[iy](y, training=training)

            for iLayer, f in enumerate(self.BranchLayersVecs[iy][iStart::]):
                if ('dropout' in f.name):
                    y = f(y, training=(training or self.BranchDropOutPredFlg))
                else:
                    y = f(y, training=training)
                if ( (self.BranchLayers_U[iy]) and (iLayer < NLayers-(1+iStart)) and (not 'dropout' in f.name) ):
                    yo = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
                    ya = tf.keras.layers.multiply([yo, y_U])
                    yb = tf.keras.layers.multiply([   y, y_V])
                    y  = tf.keras.layers.add([ya, yb])


            OutputP = Dot_Add(axes=1, n_out=self.NDot)([y, TrunkVec[iTrunk]])
            # if (self.NDotTrunk[iTrunk] == self.NDotBranch[iy]):
            #     OutputP = tf.keras.layers.Dot(axes=1)([TrunkVec[iTrunk], y])
            # elif (self.NDotTrunk[iTrunk] == self.NDotBranch[iy]-1):
            #     y1, y2  = tf.split(y, num_or_size_splits=[self.NDotTrunk[iTrunk], 1], axis=1)
            #     yDot    = tf.keras.layers.Dot(axes=1)([TrunkVec[iTrunk], y1])
            #     OutputP = tf.keras.layers.add([yDot, y2])
            # elif (self.NDotTrunk[iTrunk] == self.NDotBranch[iy]-2):
            #     y1, y2, y3 = tf.split(y, num_or_size_splits=[self.NDotTrunk[iTrunk], 1, 1], axis=1)
            #     yDot       = tf.keras.layers.Dot(axes=1)([TrunkVec[iTrunk], y1])
            #     yScaled    = tf.keras.layers.multiply([yDot, y3])
            #     OutputP    = tf.keras.layers.add([yScaled, y2])


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
        
        self.n_branch = shape1[axes[0]]
        self.n_trunk  = shape2[axes[1]]
        if (self.n_trunk < self.n_branch - 2) or (self.n_trunk > self.n_branch) or (self.n_trunk != self.n_out):
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
        if (self.n_trunk == self.n_branch):
            x2 = inputs[0]
        elif (self.n_trunk == self.n_branch-1):
            x2, x3 = tf.split(inputs[0], num_or_size_splits=[self.n_out, 1], axis=1)
        elif (self.n_trunk == self.n_branch-2):
            x2, x3, x4 = tf.split(inputs[0], num_or_size_splits=[self.n_out, 1, 1], axis=1)

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

        if (self.n_trunk == self.n_branch):
            output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True)
        elif (self.n_trunk == self.n_branch-1):
            output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True) + x3
        elif (self.n_trunk == self.n_branch-2):
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
