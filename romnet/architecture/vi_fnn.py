import numpy                       as np
import tensorflow                  as tf
import pandas                      as pd
from tensorflow.keras          import regularizers
import tensorflow_probability      as tfp

from .architecture             import Architecture

import romnet as rmnt



#===================================================================================================================================
class VI_FNN(Architecture):
    """Feed-Forward Neural Network.
    """
    
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_input, stat_output, system):
        super(VI_FNN, self).__init__()

        self.structure_name   = 'VI_FNN'
        self.structure        = InputData.structure
        self.n_hypers         = len(list(self.structure.keys()))

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
            self.sigma_like       = InputData.sigma_like
        except:
            self.sigma_like       = None

        print("\n[ROMNet - vi_fnn.py                    ]:   Constructing Variational-Inference Feed-Forward Network: ") 


        self.layers_dict      = {'All': {}}
        self.layer_names_dict = {'All': {}}

        self.system_of_components              = {}
        for system_name in list(self.structure.keys()):
            self.layers_dict[system_name]      = {}
            self.layer_names_dict[system_name] = {}

            # PCA Layers
            if (self.internal_pca_flg):
                self.layers_dict[system_name]['PCALayer']    = PCALayer(system.A, system.C, system.D)
                self.layers_dict[system_name]['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


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
                            layer_name = system_name+'-PreTransformation' + fun + '-' + str(i_trunk+1)
                            layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                            self.layers_dict[system_name]['FNN']['TransFun']      = layer
                            self.layer_names_dict[system_name]['FNN']['TransFun'] = layer_name

            
            # Main System of Components    
            System_of_Components                   = getattr(rmnt.architecture.building_blocks.system_of_components, 'FNN_System')
            self.system_of_components[system_name] = System_of_Components(InputData, system_name, self.norm_input, self.stat_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


        # Output Normalizing Layer
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

        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
                inputs_branch               = self.layers_dict[system_name]['PCALayer'](inputs_branch)
                inputs                      = tf.concat([inputs_branch, inputs_trunk], axis=1)

            hyper      = self.system_of_components[system_name].call(inputs, self.layers_dict, training=training)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0]
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
        
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0] 
                sd   = hypers_vec[1] 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.nn.relu(sd)) 
                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers) 

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
                inputs_branch               = self.layers_dict[system_name]['PCALayer'](inputs_branch)
                inputs                      = tf.concat([inputs_branch, inputs_trunk], axis=1)

            hyper      = self.system_of_components[system_name].call(inputs, self.layers_dict, training=False)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0]
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
        
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0] 
                sd   = hypers_vec[1] 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.nn.relu(sd)) 
                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers) 

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