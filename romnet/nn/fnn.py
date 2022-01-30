import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
from tensorflow.keras      import regularizers

from .nn                   import NN
from .building_blocks      import System_of_Components



#===================================================================================================================================
class FNN(NN):
    """Feed-Forward Neural Network.
    """
    
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_output):
        super(FNN, self).__init__()

        self.structure_name  = 'FNN'

        self.attention_mask  = None
        self.residual        = None


        self.input_vars      = InputData.input_vars_all
        self.n_inputs        = len(self.input_vars)


        if isinstance(InputData.output_vars, list):
            self.output_vars = InputData.output_vars
        else: 
            data_id          = list(InputData.output_vars.keys())[0]
            self.output_vars = list(pd.read_csv(InputData.PathToDataFld+'/train/'+data_id+'/'+InputData.output_vars[data_id], header=None).to_numpy()[0,:]) 
        self.n_outputs       = len(self.output_vars)

        
        if (norm_input is None):
            norm_input       = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input      = norm_input

        self.norm_output_flg = InputData.norm_output_flg


        if (stat_output is not None):
            self.output_mean  = tf.cast(stat_output['mean'][np.newaxis,...])
            self.output_std   = tf.cast(stat_output['std'][np.newaxis,...])
            self.output_min   = tf.cast(stat_output['min'][np.newaxis,...])
            self.output_max   = tf.cast(stat_output['max'][np.newaxis,...])
            self.output_range = tf.cast(self.output_max - self.output_min)
        self.stat_output = stat_output


        # Pre-Transforming Layer
        if (self.input_vars):
            self.trans_vec = []
            for ifun, fun in enumerate(self.input_vars):
                vars_list = self.trans_fun[fun]

                indxs = []
                for ivar, var in enumerate(self.input_vars):
                    if var in vars_list:
                        indxs.append(ivar)

                if (len(indxs) > 0):
                    layer_name = 'PreTransformation_' + fun
                    layer      = InputTransLayer(fun, len(self.input_vars), indxs, name=layer_name)
                    self.trans_vec.append(layer)

                  
        print("\n[ROMNet - deeponet.py               ]:   Constructing Feed-Forward Network: ") 

        self.layers_dict                 = {'FNN': {}}
        self.layer_names_dict            = {'FNN': {}}
        self.system_of_components        = {}
        self.system_of_components['FNN'] = System_of_Components(InputData, 'FNN', self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


        self.norm_output_flg           = InputData.norm_output_flg
        self.stat_output               = stat_output
        if (self.norm_output_flg) and (self.stat_output):                    
            self.output_min            = tf.constant(stat_output['min'][np.newaxis,...],  dtype=tf.keras.backend.floatx())
            self.output_max            = tf.constant(stat_output['max'][np.newaxis,...],  dtype=tf.keras.backend.floatx())
            self.output_range          = tf.constant(self.output_max - self.output_min,   dtype=tf.keras.backend.floatx())
            self.output_trans_layer    = OutputTransLayer(   self.output_range, self.output_min)
            self.output_invtrans_layer = OutputInvTransLayer(self.output_range, self.output_min)

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        y = inputs
        if (self.trans_fun):
            y = self.trans_vec[0](y)

        y     = self.system_of_components['FNN'].call(y, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        y = inputs
        if (self.trans_fun):
            y = self.trans_vec[0](y)

        y     = self.system_of_components['FNN'].call(y, training=False)

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