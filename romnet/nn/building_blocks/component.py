import numpy            as np
import tensorflow       as tf
import h5py

from .normalization import CustomNormalization
from .sub_component import Sub_Component


#=======================================================================================================================================
class Component(object):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, system_name, component_name, norm_input, layers_dict=[], layer_names_dict=[]):
        
        self.system_name           = system_name
        self.name                  = component_name
        self.layers_dict           = layers_dict
        self.layer_names_dict      = layer_names_dict

        layers_vec                 = []
        layer_names                = []

        if ('_' in self.system_name):
            self.system_type, self.system_idx = self.system_name.split('_')
            self.system_idx                   = int(self.system_idx)
        else:
            self.system_idx                   = 1
            self.system_type                  = self.system_name 

        if ('_' in self.name):
            self.type, self.idx               = self.name.split('_')
            self.idx                          = int(self.idx)
        else:
            self.type                         = self.name   
            self.idx                          = 1



        self.long_name            = self.system_name + '-' + self.name
       
        self.structure            = InputData.structure


        try:   
            self.norm_input_flg   = InputData.norm_input_flg[self.system_name][self.type]
            notfnd_flg            = False
        except:   
            self.norm_input_flg   = None
            notfnd_flg            = True
        if notfnd_flg:
            try:   
                self.norm_input_flg = InputData.norm_input_flg[self.system_name][self.name]
            except:   
                self.norm_input_flg = None


        try:   
            self.gaussnoise_rate  = InputData.gaussnoise_rate[self.system_name][self.type]
            notfnd_flg            = False
        except:   
            self.gaussnoise_rate  = None
            notfnd_flg            = True
        if notfnd_flg:
            try:   
                self.gaussnoise_rate = InputData.gaussnoise_rate[self.system_name][self.name]
            except:   
                self.gaussnoise_rate = None

    
        try:          
            self.input_vars       = InputData.input_vars[self.system_name][self.type]
        except:
            self.input_vars       = InputData.input_vars[self.system_name]


        if (norm_input is not None):
            self.norm_input       = norm_input[self.input_vars]


        try:
            self.transfered_model = InputData.transfered_model
        except:
            self.transfered_model = None


        self.call                 = self.call_classic
                
        print("[ROMNet - component.py              ]:       Constructing Component: " + self.name) 



        # Normalizing Layer
        if (self.norm_input_flg):

            layer_name        = self.long_name + '_Normalization'
        
            # if ( (BlockName == 'Trunk') and (self.PathToPODFile) ):
            #     with h5py.File(self.PathToPODFile, "r") as f:
            #         Key_       = 'NN_POD_1_Normalization'
            #         Mean       = np.array(f[Key_+'/mean:0'][:])
            #         Variance   = np.array(f[Key_+'/variance:0'][:])[...,np.newaxis]
            #         MinVals    = np.array(f[Key_+'/min_vals:0'][:])[...,np.newaxis]
            #         MaxVals    = np.array(f[Key_+'/max_vals:0'][:])[...,np.newaxis]
            #         normalizer = CustomNormalization(mean=Mean, variance=Variance, min_vals=MinVals, max_vals=MaxVals, name=layer_name)
            if (self.transfered_model is not None): 
                mean     = self.transfered_model.get_layer(layer_name).mean.numpy()
                variance = self.transfered_model.get_layer(layer_name).variance.numpy()
                min_vals = self.transfered_model.get_layer(layer_name).min_vals.numpy()
                max_vals = self.transfered_model.get_layer(layer_name).max_vals.numpy()
                layer    = CustomNormalization(mean=mean, variance=variance, min_vals=min_vals, max_vals=max_vals, name=layer_name)
            else:
                layer    = CustomNormalization(name=layer_name)
                layer.adapt(np.array(self.norm_input))
            layers_vec.append(layer)
            layer_names.append(layer_name)


        if (self.gaussnoise_rate):
            layer      = tf.keras.layers.GaussianNoise(self.gaussnoise_rate)
            layer_name = self.long_name + '_GaussNoise'
            layers_vec.append(layer)
            layer_names.append(layer_name)


        # Feed-forward Component (i.e., "Sub-Component")
        self.sub_components = {}
        for sub_component_name in self.structure[self.system_name][self.name]:

            if (sub_component_name == 'U'):
                self.call = self.call_improved                
            
            layers_dict[system_name][component_name][sub_component_name]       = []
            layers_dict[system_name][component_name][sub_component_name]      += layers_vec
            layers_vec_                                                        = layers_dict[system_name][component_name][sub_component_name]

            layer_names_dict[system_name][component_name][sub_component_name]  = []
            layer_names_dict[system_name][component_name][sub_component_name] += layer_names
            layer_names_                                                       = layer_names_dict[system_name][component_name][sub_component_name]

            self.sub_components[sub_component_name]                            = Sub_Component(InputData, self.system_name, self.name, sub_component_name, layers_vec=layers_vec_, layer_names=layer_names_)

    # ---------------------------------------------------------------------------------------------------------------------------
     

    # ---------------------------------------------------------------------------------------------------------------------------
    def call_classic(self, inputs, layers_dict, shift, stretch, training=False):

        trans_flg = False
        if ('TransFun' in layers_dict[self.system_name][self.name]):
            trans_flg = True
        #     inputs    = layers_dict[self.system_name][self.name]['TransFun'](inputs)

        if (stretch is not None):
            if (trans_flg):
                inputs = layers_dict[self.system_name][self.name]['Stretch']([inputs, tf.math.softplus(stretch)])
            else:
                inputs = layers_dict[self.system_name][self.name]['Stretch']([inputs, shift])

        if (shift   is not None):
            if (trans_flg):
                inputs = layers_dict[self.system_name][self.name]['Shift']([inputs, tf.math.softplus(shift)])
                inputs = tf.keras.layers.ReLU()(inputs) + 1.e-14
            else:
                inputs = layers_dict[self.system_name][self.name]['Shift']([inputs, shift])
            
        if (trans_flg):
            inputs = layers_dict[self.system_name][self.name]['TransFun'](inputs)

        return self.sub_components['Main'].call(inputs, training)

    # ---------------------------------------------------------------------------------------------------------------------------   


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_improved(self, inputs, layers_dict, shift, stretch, training=False):

        if (stretch is not None):
            inputs = layers_dict[self.system_name][self.name]['Stretch']([inputs, stretch])

        if (shift   is not None):
            inputs = layers_dict[self.system_name][self.name]['Shift']([inputs, shift])
            
        if ('TransFun' in layers_dict[self.system_name][self.name]):
            inputs = layers_dict[self.system_name][self.name]['TransFun'](inputs)

        y                  = inputs

        sub_component_Main = self.sub_components['Main']
        sub_component_U    = self.sub_components['U']
        sub_component_V    = self.sub_components['V']

        y_U                = sub_component_U.call(y, training=training)
        y_V                = sub_component_V.call(y, training=training)
        i_layer  = 0
        n_layers = len(sub_component_Main.layers_vec)
        while (i_layer <= n_layers-1):
            y  = sub_component_Main.call_single_layer(y, i_layer, training=training)
            if ('HL' in sub_component_Main.layer_names[i_layer]) and (i_layer < n_layers-1):
                ym = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
                yu = tf.keras.layers.multiply([ym, y_U])
                yv = tf.keras.layers.multiply([ y, y_V])
                y  = tf.keras.layers.add([yu, yv])
            i_layer += 1

        return y

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================