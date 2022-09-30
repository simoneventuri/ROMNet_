import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
import itertools
from tensorflow.python.keras              import backend as K
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.util.tf_export     import keras_export
from tensorflow.python.keras.utils        import tf_utils
from tensorflow.python.ops                import array_ops
from tensorflow.python.ops                import math_ops
import tensorflow_probability                 as tfp

from .architecture             import Architecture

import romnet as rmnt



#===================================================================================================================================
class VI_DeepONet(Architecture):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_input, stat_output, system):
        super(VI_DeepONet, self).__init__()

        self.structure_name            = 'VI_DeepONet'
        self.structure                 = InputData.structure
        self.n_hypers                  = len(list(self.structure.keys()))

        self.attention_mask            = None
        self.residual                  = None
          
          
        self.input_vars                = InputData.input_vars_all
        self.n_inputs                  = len(self.input_vars)
                     
            
        self.output_vars               = InputData.output_vars
        self.n_outputs                 = len(self.output_vars)
          
        self.branch_vars               = InputData.input_vars['DeepONetMean']['Branch']    
        self.trunk_vars                = InputData.input_vars['DeepONetMean']['Trunk']  

        self.n_branches                = len([name for name in self.structure['DeepONetMean'].keys() if 'Branch' in name])
        self.n_trunks                  = len([name for name in self.structure['DeepONetMean'].keys() if 'Trunk'  in name])

        self.n_pre_blocks              = 0
        try:
            self.n_shifts              = len([name for name in InputData.structure['DeepONetMean'].keys() if 'Shift' in name]) 
            self.n_pre_blocks         += 1 
            self.shift_vars            = InputData.input_vars[self.name]['Shift']
        except:
            self.n_shifts              = 0
        try:
            self.n_stretches           = len([name for name in InputData.structure['DeepONetMean'].keys() if 'Stretch' in name]) 
            self.n_pre_blocks         += 1
            self.stretch_vars          = InputData.input_vars[self.name]['Stretch']
        except:
            self.n_stretches           = 0
        try:
            self.n_rotations           = len([name for name in InputData.structure['DeepONetMean'].keys() if 'Rotation' in name]) 
            self.n_pre_blocks         += 1
            self.rotation_vars         = InputData.input_vars[self.name]['Rotation']
        except:
            self.n_rotations           = 0
        try:
            self.n_prenets             = len([name for name in InputData.structure['DeepONetMean'].keys() if 'PreNet' in name]) 
            self.n_pre_blocks         += 1
            self.prenet_vars           = InputData.input_vars[self.name]['PreNet']
        except:
            self.n_prenets             = 0


        try:
            self.internal_pca_flg      = InputData.internal_pca_flg
        except:
            self.internal_pca_flg      = False

        try:
            self.sigma_like            = InputData.sigma_like
        except:
            self.sigma_like            = None


        if (norm_input is None):
            norm_input                 = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input                = norm_input
        if (stat_input is not None):
            self.stat_input            = stat_input
        else:
            self.stat_input            = None

        self.trans_fun                 = InputData.trans_fun

        try:
            self.dotlayer_mult_flg     = InputData.dotlayer_mult_flg['DeepONetMean']
        except:
            self.dotlayer_mult_flg     = None
        try:
            self.dotlayer_bias_flg     = InputData.dotlayer_bias_flg['DeepONetMean']
        except:
            self.dotlayer_bias_flg     = None

        try:
            self.data_preproc_type     = InputData.data_preproc_type
        except:
            self.data_preproc_type     = None

        self.norm_output_flg           = InputData.norm_output_flg


        print("\n[ROMNet - vi_deeponet.py            ]:   Constructing Variational-Inference Deep Operator Network: ") 


        self.layers_dict      = {'All': {}}
        self.layer_names_dict = {'All': {}}

        self.system_of_components              = {}
        for system_name in list(self.structure.keys()):
            self.layers_dict[system_name]      = {}
            self.layer_names_dict[system_name] = {}


            for i_trunk in range(self.n_trunks):
                if (self.n_trunks > 1):
                    temp_str = '_'+str(i_trunk+1)
                else:
                    temp_str = ''
                self.layers_dict[system_name]['Trunk'+temp_str]      = {}
                self.layer_names_dict[system_name]['Trunk'+temp_str] = {}


            # PCA Layers
            if (self.internal_pca_flg):
                self.layers_dict[system_name]['PCALayer']    = PCALayer(system.A, system.C, system.D)
                self.layers_dict[system_name]['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


            # Pre-Transforming Layer
            if (self.trans_fun):
                for i_trunk in range(self.n_trunks):
                    if (self.n_trunks > 1):
                        temp_str = '_'+str(i_trunk+1)
                    else:
                        temp_str = ''
                    for ifun, fun in enumerate(self.trans_fun):
                        vars_list = self.trans_fun[fun]

                        indxs = []
                        for ivar, var in enumerate(self.trunk_vars):
                            if var in vars_list:
                                indxs.append(ivar)

                        if (len(indxs) > 0):
                            layer_name = system_name+'-PreTransformation' + fun + '-' + str(i_trunk+1)
                            layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                            self.layers_dict[system_name]['Trunk'+temp_str]['TransFun']      = layer
                            self.layer_names_dict[system_name]['Trunk'+temp_str]['TransFun'] = layer_name


            # Trunk-PreNets Blocks Coupling Layers
            if (self.n_pre_blocks > 0):
                for i_trunk in range(self.n_trunks):
                    if (self.n_trunks > 1):
                        temp_str = '_'+str(i_trunk+1)
                    else:
                        temp_str = ''
                    self.layers_dict[system_name]['Trunk'+temp_str]['PreNet']      = PreNet(len(self.trunk_vars))
                    self.layer_names_dict[system_name]['Trunk'+temp_str]['PreNet'] ='PreNet'


            # Main System of Components
            System_of_Components                           = getattr(rmnt.architecture.building_blocks.system_of_components, 'DeepONet_System')
            self.system_of_components[system_name]         = System_of_Components(InputData, system_name, self.norm_input, self.stat_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


            # Adding Biases to the DeepONet's Dot-Layers
            if (self.dotlayer_mult_flg):
                self.layers_dict[system_name]['MultLayer'] = MultLayer()


            # Adding Biases to the DeepONet's Dot-Layers
            if (self.dotlayer_bias_flg):
                self.layers_dict[system_name]['BiasLayer'] = BiasLayer()


        # Output Normalizing Layer
        self.norm_output_flg             = InputData.norm_output_flg
        self.stat_output                 = stat_output
        if (self.norm_output_flg) and (self.stat_output):                    
            self.output_min                                = tf.constant(stat_output['min'],  dtype=tf.keras.backend.floatx())
            self.output_max                                = tf.constant(stat_output['max'],  dtype=tf.keras.backend.floatx())
            self.output_mean                               = tf.constant(stat_output['mean'], dtype=tf.keras.backend.floatx())
            self.output_std                                = tf.constant(stat_output['std'],  dtype=tf.keras.backend.floatx())
            
            self.layers_dict['All']['OutputTrans']         = OutputTransLayer(   self.data_preproc_type, self.output_min, self.output_max, self.output_mean, self.output_std)
            self.layer_names_dict['All']['OutputTrans']    = 'OutputTrans'

            self.layers_dict['All']['OutputInvTrans']      = OutputInvTransLayer(self.data_preproc_type, self.output_min, self.output_max, self.output_mean, self.output_std)
            self.layer_names_dict['All']['OutputInvTrans'] = 'OutputInvTrans'


    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)


        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch = self.layers_dict[system_name]['PCALayer'](inputs_branch)

            hyper             = self.system_of_components[system_name].call([inputs_branch, inputs_trunk], self.layers_dict, training=training)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0] 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                # dist = tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(mu), scale_diag=self.sigma_like)
                # bij  = tfp.bijectors.ScaleMatvecDiag(scale_diag=mu)
                # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=bij)
                return dist
       
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu         = hypers_vec[0] 
                sigma_like = hypers_vec[1]
                dist       = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                # dist       = tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(mu), scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                # bij        = tfp.bijectors.ScaleMatvecDiag(scale_diag=mu)
                # dist       = tfp.distributions.TransformedDistribution(distribution=dist, bijector=bij)
                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 
        if (self.internal_pca_flg) and (self.norm_output_flg) and (self.stat_output):
            y                       = self.layers_dict['All']['OutputTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)


        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch = self.layers_dict[system_name]['PCALayer'](inputs_branch)
            hyper             = self.system_of_components[system_name].call([inputs_branch, inputs_trunk], self.layers_dict, training=False)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0]
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
        
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu         = hypers_vec[0] 
                sigma_like = hypers_vec[1] 
                #dist       = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                dist       = mu * tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(sigma_like), scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 

                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):

        input_     = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================



#=======================================================================================================================================
class AllInputTransLayer(tf.keras.layers.Layer):

    def __init__(self, f, name='AllInputTransLayer'):
        super(AllInputTransLayer, self).__init__(name=name, trainable=False)
        self.f           = f

    def call(self, inputs):
        
        if (self.f == 'log10'):
            y = tf.experimental.numpy.log10(K.maximum(inputs, K.epsilon()))
        elif (self.f == 'log'):
            y = tf.math.log(K.maximum(inputs, K.epsilon()))
        
        return y
        
#=======================================================================================================================================


#=======================================================================================================================================
class InputTransLayer(tf.keras.layers.Layer):

    def __init__(self, f, NVars, indxs, name='InputTrans'):
        super(InputTransLayer, self).__init__(name=name, trainable=False)
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
class OutputTransLayer(tf.keras.layers.Layer):

    def __init__(self, data_preproc_type, output_min, output_max, output_mean, output_std, name='OutputTrans'):
        super(OutputTransLayer, self).__init__(name=name, trainable=False)
        self.data_preproc_type = type
        self.output_min        = output_min
        self.output_max        = output_max
        self.output_mean       = output_mean
        self.output_std        = output_std
        self.output_range      = self.output_max - self.output_min

        if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
            self.call = self.call_std
        elif (self.data_preproc_type == '0to1'):
            self.call = self.call_0to1
        elif (self.data_preproc_type == 'range'):
            self.call = self.call_range
        elif (self.data_preproc_type == '-1to1'):
            self.call = self.call_m1to1
        elif (self.data_preproc_type == 'pareto'):
            self.call = self.call_pareto

    def call_std(self, inputs):
        return (inputs -  self.output_mean) / self.output_std

    def call_0to1(self, inputs):
        return (inputs -  self.output_min) / self.output_range

    def call_range(self, inputs):
        return (inputs) / self.output_range

    def call_m1to1(self, inputs):
        return 2. * (inputs - self.output_min) / (self.output_range) - 1.

    def call_pareto(self, inputs):
        return (inputs -  self.output_mean) / np.sqrt(self.output_std)
        
#=======================================================================================================================================


#=======================================================================================================================================
class OutputInvTransLayer(tf.keras.layers.Layer):

    def __init__(self, data_preproc_type, output_min, output_max, output_mean, output_std, name='OutputInvTrans'):
        super(OutputInvTransLayer, self).__init__(name=name, trainable=False)
        self.data_preproc_type = type
        self.output_min        = output_min
        self.output_max        = output_max
        self.output_mean       = output_mean
        self.output_std        = output_std
        self.output_range      = self.output_max - self.output_min
        
        if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
            self.call = self.call_std
        elif (self.data_preproc_type == '0to1'):
            self.call = self.call_0to1
        elif (self.data_preproc_type == 'range'):
            self.call = self.call_range
        elif (self.data_preproc_type == '-1to1'):
            self.call = self.call_m1to1
        elif (self.data_preproc_type == 'pareto'):
            self.call = self.call_pareto

    def call_std(self, inputs):
        return inputs * self.output_std + self.output_mean

    def call_0to1(self, inputs):
        return inputs * self.output_range + self.output_min

    def call_range(self, inputs):
        return inputs * self.output_range

    def call_m1to1(self, inputs):
        return (inputs + 1.)/2. * self.output_range + self.output_min

    def call_pareto(self, inputs):
        return inputs * np.sqrt(self.output_std) + self.output_mean

#=======================================================================================================================================



#=======================================================================================================================================
class PCALayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCALayer'):
        super(PCALayer, self).__init__(name=name, trainable=False)
        self.AT = A.T
        self.C  = C
        self.D  = D

    def call(self, inputs):
        return tf.matmul( (inputs -  self.C) / self.D, self.AT ) 
        
#=======================================================================================================================================


#=======================================================================================================================================
class PCAInvLayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCAInvLayer'):
        super(PCAInvLayer, self).__init__(name=name, trainable=False)
        self.A  = A
        self.C  = C
        self.D  = D

    def call(self, inputs):
        return tf.matmul( inputs, self.A) * self.D +  self.C
        
#=======================================================================================================================================



#=======================================================================================================================================
class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

#=======================================================================================================================================



#=======================================================================================================================================
class MultLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MultLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.stretch = self.add_weight('stretch',
                                       shape=input_shape[1:],
                                       initializer='ones',
                                       trainable=True)
    def call(self, x):
        return x * self.stretch

#=======================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.layers.PreNet')
class PreNet(_Merge):

    def __init__(self, n_y, **kwargs):
        super(PreNet, self).__init__(**kwargs)
        self.n_y              = n_y
        self.supports_masking = True


    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.
        Args:
                shape1: tuple or None. Shape of the first tensor
                shape2: tuple or None. Shape of the second tensor
        Returns:
                expected output shape when an element-wise operation is
                carried out on 2 tensors with shapes shape1 and shape2.
                tuple or None.
        Raises:
                ValueError: if shape1 and shape2 are not compatible for
                        element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError(
                            'Operands could not be broadcast '
                            'together with shapes ' + str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)



    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        self._reshape_required = False


    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = array_ops.expand_dims(x, axis=1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = array_ops.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate(
                                [x_shape[1:],
                                 array_ops.expand_dims(batch_size, axis=-1)])
                        x_transposed = array_ops.reshape(
                                x,
                                array_ops.stack(
                                        [batch_size, math_ops.reduce_prod(x_shape[1:])], axis=0))
                        x_transposed = array_ops.transpose(x_transposed, perm=(1, 0))
                        x_transposed = array_ops.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(array_ops.transpose(x, perm=dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed, we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = array_ops.shape(y)
                        y_ndim = array_ops.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([
                                array_ops.expand_dims(batch_size, axis=-1), y_shape[:y_ndim - 1]
                        ])
                        y = array_ops.reshape(y, (-1, batch_size))
                        y = array_ops.transpose(y, perm=(1, 0))
                        y = array_ops.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = array_ops.transpose(y, perm=dims)
                return y
        else:
            return self._merge_function(inputs)



    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape



    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, (tuple, list)):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, (tuple, list)):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                                             'should have the same length.')
        if all(m is None for m in mask):
            return None
        masks = [array_ops.expand_dims(m, axis=0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


    def _merge_function(self, inputs):
        # # Adding Rotations
        # a_list   = tf.split(inputs[1], num_or_size_splits=[2,1,1], axis=1)
        # x_shift  = tf.keras.layers.add([inputs[0], a_list[0]])
        # x_str    = tf.keras.layers.multiply([x_shift, a_list[1]])
    
        # x_split  = tf.split(x_str, num_or_size_splits=self.n_y, axis=1)
        # cphi     = tf.math.cos(a_list[2])
        # sphi     = tf.math.sin(a_list[2])
        # rot      = [tf.concat([cphi, sphi],  axis=1), tf.concat([sphi, -cphi], axis=1)]
        # b_list   = [tf.keras.layers.multiply([x_split[i], rot[i]]) for i in range(self.n_y)]
        # b        = tf.keras.layers.add(b_list)
        
        y_merge    = inputs[0]
        y_pre_list = inputs[1]
        y_prenet   = y_pre_list[-1]
        if (y_prenet is not None):
            y_pre_list_ = tf.split(y_prenet, num_or_size_splits=[self.n_y,1,1], axis=1)
        else:
            y_pre_list_ = y_pre_list

        y_merge    = inputs[0]
        y_pre_list = inputs[1]

        y_rotation = y_pre_list[2]
        if (y_rotation is not None):
            y_merge_split   = tf.split(y_merge, num_or_size_splits=self.n_y, axis=1)
            cphi            = tf.math.cos(y_rotation)
            sphi            = tf.math.sin(y_rotation)
            y_rotation_list = [tf.concat([cphi, -sphi],  axis=1), tf.concat([sphi, cphi], axis=1)]
            y_merge_list    = [tf.keras.layers.multiply([y_merge_split[i], y_rotation_list[i]]) for i in range(self.n_y)]
            y_merge         = tf.keras.layers.add(y_merge_list)
            
        # Adding Stretching
        y_stretch  = y_pre_list[1]
        if (y_stretch is not None):
            y_merge  = tf.keras.layers.multiply([y_merge, y_stretch])

        # Adding Shift
        y_shift    = y_pre_list[0]
        if (y_shift is not None):
            y_merge  = tf.keras.layers.add([y_merge, y_shift])

        #y_merge = tf.math.tanh(y_merge)

        return y_merge

#=======================================================================================================================================