import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
import itertools

from .nn                   import NN
from .building_blocks      import System_of_Components



#===================================================================================================================================
class DeepONet(NN):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input):
        super(DeepONet, self).__init__()

        self.structure_name  = 'DeepONet'
 
        self.attention_mask  = None
        self.residual        = None


        self.input_vars      = InputData.input_vars_all
        self.n_inputs        = len(self.input_vars)
           
  
        self.output_vars     = InputData.output_vars
        self.n_outputs       = len(self.output_vars)
  

        if (norm_input is None):
            norm_input       = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input      = norm_input

                  
        print("\n[ROMNet - deeponet.py               ]:   Constructing Deep Operator Network: ") 

        self.layers_dict                      = {'DeepONet': {}}
        self.layer_names_dict                 = {'DeepONet': {}}
        self.system_of_components             = {}
        self.system_of_components['DeepONet'] = System_of_Components(InputData, 'DeepONet', self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        # if (self.AntiPCA_flg):
        #     OutputFinal = ROM.AntiPCALayer(A0=self.A_AntiPCA, C0=self.C_AntiPCA, D0=self.D_AntiPCA)(OutputConcat)
        # else:
        #     OutputFinal = OutputConcat

        y = self.system_of_components['DeepONet'].call(inputs, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        y = self.system_of_components['DeepONet'].call(inputs, training=False)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):

        input_     = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================
