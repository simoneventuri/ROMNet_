import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
import itertools
from tensorflow.python.keras       import backend as K

from .nn                   import NN
from .building_blocks      import System_of_Components



#===================================================================================================================================
class DeepONet(NN):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_output, system):
        super(DeepONet, self).__init__()

        self.structure_name  = 'DeepONet'
 
        self.attention_mask  = None
        self.residual        = None


        self.input_vars      = InputData.input_vars_all
        self.n_inputs        = len(self.input_vars)
           
  
        self.output_vars     = InputData.output_vars
        self.n_outputs       = len(self.output_vars)

        self.branch_vars     = InputData.input_vars['DeepONet']['Branch']
        self.trunk_vars      = InputData.input_vars['DeepONet']['Trunk']  

        self.n_branches                = len([name for name in InputData.structure['DeepONet'].keys() if 'Branch' in name])
        self.n_trunks                  = len([name for name in InputData.structure['DeepONet'].keys() if 'Trunk'  in name])

        try:
            self.n_rigids              = len([name for name in InputData.structure['DeepONet'].keys() if 'Rigid' in name]) 
            self.rigid_vars            = InputData.input_vars['DeepONet']['Rigid']
        except:
            self.n_rigids              = 0

        if (norm_input is None):
            norm_input       = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input      = norm_input

        try:
            self.system_post_layer_flg = InputData.system_post_layer_flg['DeepONet']
        except:
            self.system_post_layer_flg = None

        try:
            self.internal_pca              = InputData.internal_pca
        except:
            self.internal_pca              = False

        self.norm_output_flg = InputData.norm_output_flg

        self.trans_fun       = InputData.trans_fun
  
        print("\n[ROMNet - deeponet.py               ]:   Constructing Deep Operator Network: ") 


        self.layers_dict      = {'DeepONet': {}, 'All': {}}
        for i_trunk in range(self.n_trunks):
            self.layers_dict['DeepONet']['Trunk_'+str(i_trunk+1)]      = {}

        self.layer_names_dict = {'DeepONet': {}, 'All': {}}
        for i_trunk in range(self.n_trunks):
            self.layer_names_dict['DeepONet']['Trunk_'+str(i_trunk+1)] = {}


        # PCA Layers
        if (self.internal_pca):
            self.layers_dict['DeepONet']['PCALayer']    = PCALayer(system.A, system.C, system.D)
            self.layers_dict['DeepONet']['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


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


        # Trunk-Rigid Blocks Coupling Layers
        if (self.n_rigids > 0):
            for i_trunk in range(self.n_trunks):
                self.layers_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['Shift']      = tf.keras.layers.subtract
                self.layers_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['Stretch']    = tf.keras.layers.multiply
                self.layer_names_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['Shift'] = 'Subtract'
                self.layer_names_dict['DeepONet']['Trunk_'+str(i_trunk+1)]['Shift'] = 'Multiply'


        self.system_of_components             = {}
        self.system_of_components['DeepONet'] = System_of_Components(InputData, 'DeepONet', self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


        # Adding Post Correlating / Scaling / Shifting Layer (If Needed)
        if (self.system_post_layer_flg):
            
            if (self.system_post_layer_flg == 'correlation'):
                i_branch = 0
                self.layers_dict['DeepONet']['Post']                                = system_post_layer(self.system_post_layer_flg, 'DeepONet', self.n_branches, None)

            else: 
                for i_branch in range(self.n_branches):
                    self.layers_dict['DeepONet']['Branch_'+str(i_branch+1)]['Post'] = system_post_layer(self.system_post_layer_flg, 'DeepONet', i_branch, None)

        
        # Softmax Layer
        if (self.internal_pca):
            self.layers_dict['DeepONet']['SoftMax'] = tf.keras.layers.Softmax()


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

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        if (self.internal_pca):
            inputs_branch           = self.layers_dict['DeepONet']['PCALayer'](inputs_branch)
    
        y                           = self.system_of_components['DeepONet'].call([inputs_branch, inputs_trunk], self.layers_dict, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        if (self.internal_pca):
            inputs_branch           = self.layers_dict['DeepONet']['PCALayer'](inputs_branch)

        y                           = self.system_of_components['DeepONet'].call([inputs_branch, inputs_trunk], self.layers_dict, training=False)

        if (self.norm_output_flg) and (self.stat_output):                    
            y                       = self.layers_dict['All']['OutputInvTrans'](y)

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
        return inputs * self.output_range + self.output_min
        
#=======================================================================================================================================



#=======================================================================================================================================
class PCALayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCALayer'):
        super(PCALayer, self).__init__(name=name, trainable=False)
        self.AT = A.T
        self.C  = C
        self.D  = D

        print('self.C = ', self.C)
        print('self.D = ', self.D)

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
def system_post_layer(system_post_layer_flg, system_name, i_out, transfered_model):

    if (system_post_layer_flg == 'correlation'):
        layer_name = system_name + '-Post_Correlation'
        
        if (transfered_model is not None):
            W0     = transfered_model.get_layer(layer_name).kernel.numpy()
            b0     = transfered_model.get_layer(layer_name).bias.numpy()
            W_ini  = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini  = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            W_ini  = 'he_normal'
            b_ini  = 'zeros'
        out_layer  = tf.keras.layers.Dense(units              = i_out,
                                           activation         = 'linear',
                                           use_bias           = True,
                                           kernel_initializer = W_ini,
                                           bias_initializer   = b_ini,
                                           name               = layer_name)
    

    elif (system_post_layer_flg == 'shift'):
        layer_name = system_name + '-Post_Shift_' + str(i_out+1)

        b0 = 0
        if (transfered_model is not None): 
            b0 = transfered_model.get_layer(layer_name).bias.numpy()[0]
        out_layer = bias_layer(b0=b0, layer_name=layer_name)


    else:
        layer_name = system_name + '-Post_' + system_post_layer_flg + '_' + str(i_out+1)
        
        if (transfered_model is not None):
            W0     = transfered_model.get_layer(layer_name).kernel.numpy()
            b0     = transfered_model.get_layer(layer_name).bias.numpy()
            W_ini  = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini  = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            W_ini  = 'he_normal'
            b_ini  = 'zeros'
        out_layer  = tf.keras.layers.Dense(units              = 1,
                                           activation         = system_post_layer_flg,
                                           use_bias           = True,
                                           kernel_initializer = W_ini,
                                           bias_initializer   = b_ini,
                                           name               = layer_name)

    return out_layer

#=======================================================================================================================================