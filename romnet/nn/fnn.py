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
    def __init__(self, InputData, norm_input):
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

                  
        print("\n[ROMNet - deeponet.py               ]:   Constructing Feed-Forward Network: ") 

        self.layers_dict                 = {'FNN': {}}
        self.layer_names_dict            = {'FNN': {}}
        self.system_of_components        = {}
        self.system_of_components['FNN'] = System_of_Components(InputData, 'FNN', self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        # if (self.AntiPCA_flg):
        #     OutputFinal = ROM.AntiPCALayer(A0=self.A_AntiPCA, C0=self.C_AntiPCA, D0=self.D_AntiPCA)(OutputConcat)
        # else:
        #     OutputFinal = OutputConcat

        y        = self.system_of_components['FNN'].call(inputs, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):
        input_  = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================
