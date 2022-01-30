import numpy                    as np
import tensorflow               as tf

from tensorflow.keras       import regularizers
from tensorflow.keras       import activations
from tensorflow.keras       import initializers

from ...training            import L1L2Regularizer


#=======================================================================================================================================
class Sub_Component(object):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, system_name, component_name, sub_component_name, layers_vec=[], layer_names=[]):
        
        self.system_name          = system_name
        self.component_name       = component_name
        self.name                 = sub_component_name
        self.layers_vec           = layers_vec
        self.layer_names          = layer_names          

        if ('_' in self.system_name):
            self.system_type, self.system_idx       = self.system_name.split('_')
            self.system_idx                         = int(self.system_idx)
        else:
            self.system_idx                         = 1
            self.system_type                        = self.system_name 

        if ('_' in self.component_name):
            self.component_type, self.component_idx = self.component_name.split('_')
            self.component_idx                      = int(self.component_idx)
        else:
            self.component_type                     = self.component_name   
            self.component_idx                      = 1

        self.long_name            = self.system_name + '-' + self.component_name + '-' + self.name


        try:
            self.n_neurons        = InputData.n_neurons[self.system_name][self.component_type][self.name]
            self.act_funcs        = InputData.act_funcs[self.system_name][self.component_type][self.name]
        except:
            self.n_neurons        = InputData.n_neurons[self.system_name][self.component_name][self.name]
            self.act_funcs        = InputData.act_funcs[self.system_name][self.component_name][self.name]
        self.n_layers             = len(self.n_neurons)


        try:
            self.trainable_flg    = InputData.trainable_flg[self.system_name]
            notfnd_flg            = False
        except:
            self.trainable_flg    = 'all'
            notfnd_flg            = True
        if notfnd_flg:
            try:
                self.trainable_flg    = InputData.trainable_flg[self.system_name][self.component_type]
                notfnd_flg            = False
            except:
                self.trainable_flg    = 'all'
                notfnd_flg            = True
        if notfnd_flg:
            try:
                self.trainable_flg    = InputData.trainable_flg[self.system_name][self.component_type][self.name]
            except:
                self.trainable_flg    = 'all'


        try:
            self.dropout_rate     = InputData.dropout_rate[self.system_name][self.component_type][self.name]
            notfnd_flg            = False
        except:
            self.dropout_rate     = None 
            notfnd_flg            = True
        if notfnd_flg:
            try:
                self.dropout_rate     = InputData.dropout_rate[self.system_name][self.component_name][self.name]
            except:
                self.dropout_rate     = None 


        try:
            self.dropout_pred_flg = InputData.dropout_pred_flg[self.system_name][self.component_type][self.name]
            notfnd_flg            = False
        except:
            self.dropout_pred_flg = False
            notfnd_flg            = True
        if notfnd_flg:
            try:
                self.dropout_pred_flg = InputData.dropout_pred_flg[self.system_name][self.component_name][self.name]
            except:
                self.dropout_pred_flg = False


        try:
            self.softmax_flg      = InputData.softmax_flg[self.system_name][self.component_type][self.name]
            notfnd_flg            = False
        except:
            self.softmax_flg      = None
            notfnd_flg            = True
        if notfnd_flg:
            try:
                self.softmax_flg      = InputData.softmax_flg[self.system_name][self.component_name][self.name]
            except:
                self.softmax_flg      = None


        self.weight_decay_coeffs  = InputData.weight_decay_coeffs
        
        try:
            self.transfered_model = InputData.transfered_model
        except:
            self.transfered_model = None



        ### Weights L1 and L2 Regularization Coefficients 
        kW1     = self.weight_decay_coeffs[0]
        kW2     = self.weight_decay_coeffs[1]

        for i_layer in range(self.n_layers):
            layer_name = self.long_name + '-HL_' + str(i_layer+1)
            n_neurons  = self.n_neurons[i_layer]
            act_fun    = self.act_funcs[i_layer]
            self.layer_names.append(layer_name)


            # Parameters Initialization
            ### Biases L1 and L2 Regularization Coefficients 
            if (i_layer < self.n_layers-1):
                if (len(self.weight_decay_coeffs) == 2):
                    kb1 = kW1
                    kb2 = kW2
                else:
                    kb1 = self.weight_decay_coeffs[2]
                    kb2 = self.weight_decay_coeffs[3]
            else:
                kb1 = 0.
                kb2 = 0.

            if (self.transfered_model is not None):
                W0    = self.transfered_model.get_layer(layer_name).kernel.numpy()
                b0    = self.transfered_model.get_layer(layer_name).bias.numpy()
                W_ini = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
                b_ini = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                W_reg = L1L2Regularizer(kW1, kW2, W0)
                b_reg = L1L2Regularizer(kb1, kb2, b0)
            else:
                W_ini = 'he_normal' if (act_fun == 'relu') else 'glorot_normal'
                b_ini = 'zeros'
                W_reg = regularizers.l1_l2(l1=kW1, l2=kW2)
                b_reg = regularizers.l1_l2(l1=kb1, l2=kb2)

            # Constructing Kera Layer
            layer = tf.keras.layers.Dense(units              = n_neurons,
                                          activation         = act_fun,
                                          use_bias           = True,
                                          kernel_initializer = W_ini,
                                          bias_initializer   = b_ini,
                                          kernel_regularizer = W_reg,
                                          bias_regularizer   = b_reg,
                                          name               = layer_name)


            # Trainable Layer?
            if (self.trainable_flg == 'none'):
                layer.trainable = False
            elif (self.trainable_flg == 'only_last'):
                if (i_layer < self.n_layers-1):
                    layer.trainable = False


            # Add Layer to the List
            self.layers_vec.append(layer)


            # Adding Dropout if Needed
            if (i_layer < self.n_layers-1) and (self.dropout_rate is not None):

                layer_name              = self.long_name + '-Dropout_' + str(i_layer+1)
                self.layer_names.append(layer_name)
                dropout_layer           = tf.keras.layers.Dropout(self.dropout_rate, input_shape=(n_neurons,))
                dropout_layer.trainable = self.dropout_pred_flg
                self.layers_vec.append( dropout_layer )
        

        # Adding SoftMax if Needed
        if (self.softmax_flg):
            layer_name = self.long_name + '-SoftMax'
            self.layer_names.append(layer_name)
            self.layers_vec.append(tf.keras.layers.Softmax())
    

        self.n_layers = len(self.layers_vec)
        print("[ROMNet - sub_component.py          ]:         Constructed Sub-Component: " + self.name + " with Layers:      ", self.layers_vec) 
        #print("[ROMNet - sub_component.py          ]:         Constructed Sub-Component: " + self.name + " with Layer Names: ", self.layer_names) 

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_single_layer(self, inputs, i_layer, training=False):

        y = self.layers_vec[i_layer](inputs, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        y = inputs        
        for i_layer in range(self.n_layers):
            y = self.layers_vec[i_layer](y, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================