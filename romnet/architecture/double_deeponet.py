import numpy                     as np
import tensorflow                as tf
import pandas                    as pd
import itertools
from tensorflow.python.keras import backend
from tensorflow.python.ops   import array_ops

from .architecture           import Architecture

import romnet as rmnt



#===================================================================================================================================
class Double_DeepONet(Architecture):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_input, stat_output, system):
        super(Double_DeepONet, self).__init__()

        self.structure_name  = 'Double_DeepONet'
 
        self.attention_mask  = None
        self.residual        = None


        self.input_vars      = InputData.input_vars_all
        self.n_inputs        = len(self.input_vars)

        self.output_vars     = InputData.output_vars
        self.n_outputs       = len(self.output_vars)
           
        self.branch_vars     = InputData.input_vars['DeepONet']['Branch']
        self.trunk_vars      = InputData.input_vars['DeepONet']['Trunk']

        self.n_trunks                  = len([name for name in InputData.structure['DeepONet'].keys() if 'Trunk' in name])

        try:
            self.n_rigids              = len([name for name in InputData.structure['DeepONet'].keys() if 'Rigid' in name]) 
            self.rigid_vars            = InputData.input_vars['DeepONet']['Rigid']
        except:
            self.n_rigids              = 0
  
        if (norm_input is None):
            norm_input       = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input      = norm_input
        if (stat_input is not None):
            self.stat_input                = stat_input
        else:
            self.stat_input                = None

        self.t_scale            = InputData.t_scale
        self.n_deeponets        = InputData.n_deeponets
        self.trans_fun          = InputData.trans_fun
        if (self.n_deeponets == 1):
            self.deeponet_names  = ['DeepONet'] 
            self.i_trans_2       = 0
            self.deeponet_2_name = 'DeepONet'
        elif (self.n_deeponets == 2):
            self.deeponet_names  = ['DeepONet', 'DeepONet_2'] 
            self.i_trans_2       = 1
            self.deeponet_2_name = 'DeepONet_2'

        print("\n[ROMNet - double_deeponet.py        ]:   Constructing Two Deep Operator Networks in Series: ") 


        self.layers_dict      = {'All': {}}
        self.layer_names_dict = {'All': {}}
        for deeponet_name in self.deeponet_names:

            self.layers_dict[deeponet_name] = {'All': {}}
            for i_trunk in range(self.n_trunks):
                self.layers_dict[deeponet_name]['Trunk_'+str(i_trunk+1)]      = {}

            self.layer_names_dict[deeponet_name] = {'All': {}}
            for i_trunk in range(self.n_trunks):
                self.layer_names_dict[deeponet_name]['Trunk_'+str(i_trunk+1)] = {}


            # Trunk-Rigid Blocks Coupling Layers
            if (self.n_rigids > 0):
                for i_trunk in range(self.n_trunks):
                    self.layers_dict[deeponet_name]['Trunk_'+str(i_trunk+1)]['Shift']      = tf.keras.layers.subtract
                    self.layers_dict[deeponet_name]['Trunk_'+str(i_trunk+1)]['Stretch']    = tf.keras.layers.multiply
                    self.layer_names_dict[deeponet_name]['Trunk_'+str(i_trunk+1)]['Shift'] = 'Subtract'
                    self.layer_names_dict[deeponet_name]['Trunk_'+str(i_trunk+1)]['Shift'] = 'Multiply'


        # Pre-Transforming Layer
        if (self.trans_fun):
            for i_trunk in range(self.n_trunks):
                for ifun, fun in enumerate(self.trans_fun):
                    vars_list = self.trans_fun[fun]

                    indxs = []
                    for ivar, var in enumerate(self.trunk_vars):
                        if var in vars_list:
                            indxs.append(ivar)

                    if (len(indxs) > 0):
                        layer_name = 'PreTransformation' + fun + '-' + str(i_trunk+1)
                        layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                        self.layers_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['TransFun']      = layer
                        self.layer_names_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['TransFun'] = layer_name


        self.layers_dict['DeepONet']['All']['Randomize']      = RandomLayer(x_scale=self.t_scale, name='Randomize')
        self.layer_names_dict['DeepONet']['All']['Randomize'] = 'Randomize'

        self.layers_dict['DeepONet']['All']['Subtract']      = tf.keras.layers.subtract
        self.layer_names_dict['DeepONet']['All']['Subtract'] = 'Subtract'


        self.system_of_components                    = {}
        for deeponet_name in self.deeponet_names:
            System_of_Components                     = getattr(rmnt.architecture.building_blocks.system_of_components, 'DeepONet_System')
            self.system_of_components[deeponet_name] = System_of_Components(InputData, deeponet_name, self.norm_input, self.stat_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)
            

        self.norm_output_flg             = InputData.norm_output_flg
        self.stat_output                 = stat_output
        if (self.norm_output_flg) and (self.stat_output):                    
            self.output_min                                = tf.constant(stat_output['min'],  dtype=tf.keras.backend.floatx())
            self.output_max                                = tf.constant(stat_output['max'],  dtype=tf.keras.backend.floatx())
            self.output_range                              = tf.constant(self.output_max - self.output_min,   dtype=tf.keras.backend.floatx())
            
            self.layers_dict['All']['OutputTrans']         = OutputTransLayer(   self.output_range, self.output_min)
            self.layer_names_dict['All']['OutputTrans']    = 'OutputTrans'

            self.layers_dict['All']['OutputInvTrans']      = OutputInvTransLayer(self.output_range, self.output_min)
            self.layer_names_dict['All']['OutputInvTrans'] = 'OutputInvTrans'

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        inputs_branch, inputs_trunk    = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
        
        inputs_trunk_1                 = self.layers_dict['DeepONet']['All']['Randomize'](inputs_trunk)
        inputs_trunk_2                 = self.layers_dict['DeepONet']['All']['Subtract']([inputs_trunk, inputs_trunk_1])

        inputs_1                       = [inputs_branch, inputs_trunk_1]
        y_1                            = self.system_of_components['DeepONet'].call(inputs_1, self.layers_dict, training=training)

        if (self.norm_output_flg) and (self.stat_output):                    
            y_1                        = self.layers_dict['All']['OutputInvTrans'](y_1)

        inputs_2                       = [y_1, inputs_trunk_2]
        y_2                            = self.system_of_components[self.deeponet_2_name].call(inputs_2, self.layers_dict, training=training)

        return y_2

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict_deeponet_1(self, inputs):

        inputs_1       = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        y              = self.system_of_components['DeepONet'].call(inputs_1, self.layers_dict, training=False)

        if (self.norm_output_flg) and (self.stat_output):                    
            y            = self.layers_dict['All']['OutputInvTrans'](y)
        
        return y.numpy()

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict_deeponet_2(self, inputs):

        inputs_1       = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        y              = self.system_of_components[self.deeponet_2_name].call(inputs_1, self.layers_dict, training=False)

        if (self.norm_output_flg) and (self.stat_output):                    
            y           = self.layers_dict['All']['OutputInvTrans'](y)

        return y.numpy()

    # ---------------------------------------------------------------------------------------------------------------------------


    # # ---------------------------------------------------------------------------------------------------------------------------
    # def call_test(self, inputs, training=False):

    #     inputs_branch, inputs_trunk    = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
        
    #     inputs_trunk_1                 = self.rand_layer(inputs_trunk)
    #     inputs_trunk_2                 = tf.keras.layers.subtract([inputs_trunk, inputs_trunk_1])
        
    #     if (self.trans_fun):
    #         inputs_trunk_1_log = self.trans_dicts['DeepONet'](inputs_trunk_1)

    #     inputs_1         = [inputs_branch, inputs_trunk_1_log]
    #     y_1              = self.system_of_components['DeepONet'].call(inputs_1, training=training)

    #     if (self.norm_output_flg) and (self.stat_output):                    
    #         y_1          = self.output_invtrans_layer(y_1)
 
    #     if (self.trans_fun):
    #         inputs_trunk_2_log = self.trans_dicts[self.deeponet_2_name](inputs_trunk_2)

    #     inputs_2         = [y_1, inputs_trunk_2_log]
    #     y_2              = self.system_of_components[self.deeponet_2_name].call(inputs_2, training=training)

    #     return inputs_trunk_1_log.numpy(), inputs_trunk_2_log.numpy(), y_1.numpy(), y_2.numpy()

    # # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):

        input_     = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================



#=======================================================================================================================================
class RandomLayer(tf.keras.layers.Layer):

    def __init__(self, x_scale='lin', x_min=1.e-14, min_val=0., max_val=1., seed=3, name='RandomLayer'):
        super(RandomLayer, self).__init__(name=name, trainable=False)
        
        self.x_scale = x_scale
        self.x_min   = x_min
        self.min_val = min_val
        self.max_val = max_val
        self.seed    = seed


    def build(self, input_shape):
        
        if   (self.x_scale == 'lin'):
            self.call = self.call_lin
        elif (self.x_scale == 'log'):
            self.call = self.call_log


    def call_lin(self, inputs, training=False):

        rand_vec = backend.random_uniform(shape=array_ops.shape(inputs), minval=self.min_val, maxval=self.max_val, dtype=self.dtype, seed=self.seed)

        y_1      = tf.multiply(inputs, rand_vec)

        #y_2      = tf.subtract(inputs, y_1)

        return y_1#, y_2 


    def call_log(self, inputs, training=False):

        rand_vec = backend.random_uniform(shape=array_ops.shape(inputs), minval=0., maxval=1., dtype=self.dtype, seed=self.seed)
        rand_vec = 10.**( 8.*rand_vec - 8.)

        y_1      = tf.multiply(inputs, rand_vec)

        #y_2      = tf.subtract(inputs, y_1)

        return y_1#, y_2


        
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

    def __init__(self, output_range, output_min, name='OutputTrans'):
        super(OutputTransLayer, self).__init__(name=name, trainable=False)
        self.output_range = output_range
        self.output_min   = output_min

    def call(self, inputs):
        return (inputs -  self.output_min) / self.output_range
        
#=======================================================================================================================================



#=======================================================================================================================================
class OutputInvTransLayer(tf.keras.layers.Layer):

    def __init__(self, output_range, output_min, name='OutputInvTrans'):
        super(OutputInvTransLayer, self).__init__(name=name, trainable=False)
        self.output_range = output_range
        self.output_min   = output_min

    def call(self, inputs):
        y = inputs * self.output_range + self.output_min
        return y
        
#=======================================================================================================================================