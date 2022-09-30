import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
from tensorflow.keras      import regularizers

from .architecture         import Architecture

import romnet as rmnt



#===================================================================================================================================
class Autoencoder(Architecture):
    """Feed-Forward Neural Network.
    """
    
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_input, stat_output, system):
        super(Autoencoder, self).__init__()

        self.structure_name   = 'Autoencoder'
 
        self.attention_mask   = None
        self.residual         = None
 
 
        self.input_vars       = InputData.input_vars_all
        self.n_inputs         = len(self.input_vars)
 
 
        self.output_vars      = InputData.output_vars
        self.n_outputs        = len(self.output_vars)
 
        
        if (norm_input is None):
            norm_input        = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input       = norm_input
        if (stat_input is not None):
            self.stat_input   = stat_input
        else:
            self.stat_input   = None

        self.norm_output_flg  = InputData.norm_output_flg
 
        self.trans_fun        = InputData.trans_fun

        try:
            self.internal_pca_flg = InputData.internal_pca_flg
        except:
            self.internal_pca_flg = False

        try:
            self.data_preproc_type     = InputData.data_preproc_type
        except:
            self.data_preproc_type     = None

        print("\n[ROMNet - autoencoder.py            ]:   Constructing Autoencoder: ") 


        self.layers_dict      = {'Encoder': {}, 'Decoder': {}, 'All': {}}
        self.layer_names_dict = {'Encoder': {}, 'Decoder': {}, 'All': {}}


        # PCA Layers
        if (self.internal_pca_flg):
            self.layers_dict['Encoder']['PCALayer']    = PCALayer(system.A, system.C, system.D)
            self.layers_dict['Decoder']['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


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
                        layer_name = 'Encoder-PreTransformation' + fun + '-' + str(i_trunk+1)
                        layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                        self.layers_dict['Encoder']['FNN']['TransFun']      = layer
                        self.layer_names_dict['Encoder']['FNN']['TransFun'] = layer_name

        
        # Main System of Components    
        self.system_of_components            = {}
        System_of_Components_Encoder         = getattr(rmnt.architecture.building_blocks.system_of_components, 'FNN_System')
        self.system_of_components['Encoder'] = System_of_Components_Encoder(InputData, 'Encoder', self.norm_input, self.stat_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)
        System_of_Components_Decoder         = getattr(rmnt.architecture.building_blocks.system_of_components, 'FNN_System')
        self.system_of_components['Decoder'] = System_of_Components_Decoder(InputData, 'Decoder', None,            None,            layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


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

        # if (self.internal_pca_flg):
        #     inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
        #     inputs_branch               = self.layers_dict[system_name]['PCALayer'](inputs_branch)
        #     inputs                      = tf.concat([inputs_branch, inputs_trunk], axis=1)

        y = self.system_of_components['Encoder'].call(inputs, self.layers_dict, training=training)
        y = self.system_of_components['Decoder'].call(y,      self.layers_dict, training=training)

        if (self.norm_output_flg) and (self.stat_output):   
            y = self.layers_dict['All']['OutputTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        # if (self.internal_pca_flg):
        #     inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
        #     inputs_branch               = self.layers_dict[system_name]['PCALayer'](inputs_branch)
        #     inputs                      = tf.concat([inputs_branch, inputs_trunk], axis=1)
            
        y = self.system_of_components['Encoder'].call(inputs, self.layers_dict, training=training)
        y = self.system_of_components['Decoder'].call(y,      self.layers_dict, training=training)
        
        if (self.norm_output_flg) and (self.stat_output):                    
            y = self.layers_dict['All']['OutputInvTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):
        input_  = tf.keras.Input(shape=[self.n_inputs,])
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
        self.data_preproc_type = data_preproc_type
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

    def call_std(self, inputs, training=False):
        return (inputs -  self.output_mean) / self.output_std

    def call_0to1(self, inputs, training=False):
        return (inputs -  self.output_min) / self.output_range

    def call_range(self, inputs, training=False):
        return (inputs) / self.output_range

    def call_m1to1(self, inputs, training=False):
        return 2. * (inputs - self.output_min) / (self.output_range) - 1.

    def call_pareto(self, inputs, training=False):
        return (inputs -  self.output_mean) / np.sqrt(self.output_std)
        
#=======================================================================================================================================


#=======================================================================================================================================
class OutputInvTransLayer(tf.keras.layers.Layer):

    def __init__(self, data_preproc_type, output_min, output_max, output_mean, output_std, name='OutputInvTrans'):
        super(OutputInvTransLayer, self).__init__(name=name, trainable=False)
        self.data_preproc_type = data_preproc_type
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

    def call_std(self, inputs, training=False):
        return inputs * self.output_std + self.output_mean

    def call_0to1(self, inputs, training=False):
        return inputs * self.output_range + self.output_min

    def call_range(self, inputs, training=False):
        return inputs * self.output_range

    def call_m1to1(self, inputs, training=False):
        return (inputs + 1.)/2. * self.output_range + self.output_min

    def call_pareto(self, inputs, training=False):
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