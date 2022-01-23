import numpy                     as np
import tensorflow                as tf
import pandas                    as pd
import itertools
from tensorflow.python.keras import backend
from tensorflow.python.ops   import array_ops

from .nn                     import NN
from .building_blocks        import System_of_Components


#===================================================================================================================================
class Double_DeepONet(NN):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input):
        super(Double_DeepONet, self).__init__()

        self.structure_name  = 'Double_DeepONet'
 
        self.attention_mask  = None
        self.residual        = None


        self.input_vars      = InputData.input_vars_all
        self.n_inputs        = len(self.input_vars)
           
        self.branch_vars     = InputData.input_vars['DeepONet']['Branch']
        self.trunk_vars      = InputData.input_vars['DeepONet']['Trunk']
  
        self.output_vars     = InputData.output_vars
        self.n_outputs       = len(self.output_vars)
  

        if (norm_input is None):
            norm_input       = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input      = norm_input


        self.t_scale         = InputData.t_scale
        self.deeponet_2_name = 'DeepONet' #'DeepONet_2'
                  
        print("\n[ROMNet - double_deeponet.py        ]:   Constructing Two Deep Operator Networks in Series: ") 

        self.layers_dict                                = {'DeepONet': {}}
        self.layer_names_dict                           = {'DeepONet': {}}
        self.system_of_components                       = {}
        self.system_of_components['DeepONet']           = System_of_Components(InputData, 'DeepONet',   self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)
        
        if (self.deeponet_2_name != 'DeepONet'):
            self.layers_dict[self.deeponet_2_name]          = {}
            self.layer_names_dict[self.deeponet_2_name]     = {}
            self.system_of_components[self.deeponet_2_name] = System_of_Components(InputData, self.deeponet_2_name, self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        inputs_trunk_1 = RandomLayer(x_scale=self.t_scale, name='RandomLayer')(inputs_trunk)
        inputs_trunk_2 = tf.keras.layers.subtract([inputs_trunk, inputs_trunk_1]) 

        inputs_1       = tf.keras.layers.Concatenate(axis=1)([inputs_branch, inputs_trunk_1])
        y_1            = self.system_of_components['DeepONet'].call(inputs_1, training=training)
         
        inputs_2       = tf.keras.layers.Concatenate(axis=1)([y_1, inputs_trunk_2])
        y_2            = self.system_of_components[self.deeponet_2_name].call(inputs_2, training=training)

        #print("\n[ROMNet - double_deeponet.py        ]:   inputs_trunk = ", inputs_trunk) 
        #print("\n[ROMNet - double_deeponet.py        ]:   inputs_trunk_1 = ", inputs_trunk_1) 
        #print("\n[ROMNet - double_deeponet.py        ]:   inputs_trunk_2 = ", inputs_trunk_2) 
        #print("\n[ROMNet - double_deeponet.py        ]:   inputs_1 = ", inputs_1) 
        #print("\n[ROMNet - double_deeponet.py        ]:   inputs_2 = ", inputs_2) 

        return y_2

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        y = self.system_of_components[self.deeponet_2_name].call(inputs, training=False)

        return -y

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):

        input_     = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================


#=======================================================================================================================================
class RandomLayer(tf.keras.layers.Layer):

    def __init__(self, x_scale='lin', x_min=1.e-12, min_val=0., max_val=1., seed=None, name='RandomLayer'):
        super(RandomLayer, self).__init__(name=name, trainable=False)
        
        self.x_scale = x_scale
        self.x_min   = x_min
        self.min_val = min_val
        self.max_val = max_val
        self.seed    = seed


    def build(self, input_shape):

        
        self.n_x = 1#tf.TensorShape(input_shape)[0]
        
        if   (self.x_scale == 'lin'):
            self.call = self.call_lin
        elif (self.x_scale == 'log'):
            self.call = self.call_log


    def call_lin(self, inputs, training=False):

        rand_vec = backend.random_uniform(shape=array_ops.shape(inputs), minval=self.min_val, maxval=self.max_val, dtype=self.dtype, seed=self.seed)
        tf.print('rand_vec = ', rand_vec)

        return tf.multiply(inputs, rand_vec)


    def call_log(self, inputs, training=False):

        rand_vec = backend.random_uniform(shape=array_ops.shape(inputs), minval=self.min_val, maxval=self.max_val, dtype=self.dtype, seed=self.seed)

        y        = tf.math.exp( tf.multiply(rand_vec, (tf.math.log(inputs) - tf.math.log(self.x_min)) ) + tf.math.log(self.x_min) )
        
        return y


        
#=======================================================================================================================================
