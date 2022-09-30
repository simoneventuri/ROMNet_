import numpy                    as np
import tensorflow               as tf

from tensorflow.keras       import regularizers
from tensorflow.keras       import activations
from tensorflow.keras       import initializers

from ...training            import L1L2Regularizer
from .layer                 import Layer


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
            self.layer_type       = InputData.layer_type[self.system_name][self.component_type][self.name]
            notfnd_flg            = False
        except:
            try:
                self.layer_type   = InputData.layer_type[self.system_name][self.component_name][self.name]
            except:
                self.layer_type   = ['TF']*self.n_layers


        notfnd_flg = True
        try:
            self.trainable_flg    = InputData.trainable_flg[self.system_name][self.component_type][self.name]
            notfnd_flg            = not isinstance(self.trainable_flg, str)
        except:
            pass
        if notfnd_flg:
            try:
                self.trainable_flg = InputData.trainable_flg[self.system_name][self.component_type]
                notfnd_flg         = not isinstance(self.trainable_flg, str)
            except:
                pass
        if notfnd_flg:
            try:
                self.trainable_flg = InputData.trainable_flg[self.system_name]
                notfnd_flg         = not isinstance(self.trainable_flg, str)
            except:
                pass
        if (notfnd_flg):
            self.trainable_flg = 'all'
            notfnd_flg         = True 


        try:
            self.dropout_rate      = InputData.dropout_rate[self.system_name][self.component_type][self.name]
            notfnd_flg             = False
        except:
            self.dropout_rate      = None 
            notfnd_flg             = True
        if notfnd_flg:
            try:
                self.dropout_rate  = InputData.dropout_rate[self.system_name][self.component_name][self.name]
            except:
                self.dropout_rate  = None 

        try:
            self.dropout_pred_flg  = InputData.dropout_pred_flg[self.system_name][self.component_type][self.name]
            notfnd_flg             = False
        except:
            self.dropout_pred_flg  = False
            notfnd_flg             = True
        if notfnd_flg:
            try:
                self.dropout_pred_flg = InputData.dropout_pred_flg[self.system_name][self.component_name][self.name]
            except:
                self.dropout_pred_flg = False


        try:
            self.softmax_flg       = InputData.softmax_flg[self.system_name][self.component_type][self.name]
            notfnd_flg             = False
        except:
            self.softmax_flg       = None
            notfnd_flg             = True
        if notfnd_flg:
            try:
                self.softmax_flg   = InputData.softmax_flg[self.system_name][self.component_name][self.name]
            except:
                self.softmax_flg   = None


        notfnd_flg                 = True
        try:
            self.reg_coeffs        = InputData.reg_coeffs[self.system_name][self.component_type][self.name]
            notfnd_flg             = not isinstance(self.reg_coeffs, list)
        except:
            pass
        if (notfnd_flg):
            try:
                self.reg_coeffs    = InputData.reg_coeffs[self.system_name][self.component_name]
                notfnd_flg         = not isinstance(self.reg_coeffs, list)
            except:
                pass
        if (notfnd_flg):
            try:
                self.reg_coeffs    = InputData.reg_coeffs[self.system_name][self.component_type]
                notfnd_flg         = not isinstance(self.reg_coeffs, list)
            except:
                pass    
        if (notfnd_flg):
            try:
                self.reg_coeffs    = InputData.reg_coeffs[self.system_name]
                notfnd_flg         = not isinstance(self.reg_coeffs, list)
            except:
                pass
        if (notfnd_flg):
            self.reg_coeffs        = InputData.weight_decay_coeffs
        

        try:
            self.transfered_model  = InputData.transfered_model
        except:
            self.transfered_model  = None



        for i_layer in range(self.n_layers):
            layer_name = self.long_name + '-HL_' + str(i_layer+1)
            self.layer_names.append(layer_name)

            # Adding Layer
            layer      =  Layer(InputData, 
                                layer_type          = self.layer_type[i_layer],
                                i_layer             = i_layer, 
                                n_layers            = self.n_layers, 
                                layer_name          = layer_name, 
                                n_neurons           = self.n_neurons[i_layer], 
                                act_func            = self.act_funcs[i_layer], 
                                use_bias            = True, 
                                trainable_flg       = self.trainable_flg, 
                                reg_coeffs          = self.reg_coeffs,
                                transfered_model    = self.transfered_model)
            self.layers_vec.append(layer.build())


            # Adding Dropout if Needed
            if (i_layer < self.n_layers-1) and (self.dropout_rate is not None):

                layer_name              = self.long_name + '-Dropout_' + str(i_layer+1)
                self.layer_names.append(layer_name)
                dropout_layer           = tf.keras.layers.Dropout(self.dropout_rate, input_shape=(self.n_neurons[i_layer],))
                dropout_layer.trainable = self.dropout_pred_flg
                self.layers_vec.append( dropout_layer )
        

        # Adding SoftMax if Needed
        if (self.softmax_flg is True):
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
            training_ = training
            if ('Dropout' in self.layer_names[i_layer]) and (training == False):
                training_ = self.layers_vec[i_layer].trainable
            y = self.layers_vec[i_layer](y, training=training_)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================