import numpy            as np
import tensorflow       as tf
import h5py

from .normalization import CustomNormalization
from .sub_component import Sub_Component


#=======================================================================================================================================
class Component(object):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, system_name, component_name, norm_input, trans_vec=[], trans_names=[], norm_vec=[], norm_names=[], layers_dict=[], layer_names=[]):
        
        self.system_name           = system_name
        self.name                  = component_name

        self.trans_vec             = trans_vec
        self.trans_names           = trans_names
        self.norm_vec              = norm_vec
        self.norm_names            = norm_names
        self.layers_dict           = layers_dict
        self.layer_names           = layer_names

        layers_vec                 = []

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
        except:   
            self.norm_input_flg   = None
        try:   
            self.norm_input_flg   = InputData.norm_input_flg[self.system_name][self.name]
        except:   
            self.norm_input_flg   = None

    
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

        try:    
            self.trans_fun        = InputData.trans_fun
        except:       
            self.trans_fun        = None


        self.call                 = self.call_classic
                
        print("[ROMNet - component.py              ]:       Constructing Component: " + self.name) 


        # Pre-Transforming Layer
        if (self.trans_fun):
            for ifun, fun in enumerate(self.trans_fun):
                vars_list = self.trans_fun[fun]

                indxs = []
                for ivar, var in enumerate(self.input_vars):
                    if var in vars_list:
                        indxs.append(ivar)

                if (len(indxs) > 0):
                    layer      = TransLayer(fun, len(self.input_vars), indxs, name=layer_name)
                    layer_name = self.long_name + '-PreTransformation_' + fun
                    layers_vec.append(layer)
                    layer_names.append(layer_name)
                    self.trans_vec.append(layer)
                    self.trans_names.append(layer_name)
            

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
            self.norm_vec.append(layer)
            self.norm_names.append(layer_name)


        # Feed-forward Component (i.e., "Sub-Component")
        self.sub_components = {}
        for sub_component_name in self.structure[self.system_name][self.name]:

            if (sub_component_name == 'U'):
                self.call = self.call_improved                
            
            layers_dict[system_name][component_name][sub_component_name]  = []
            layers_dict[system_name][component_name][sub_component_name] += layers_vec
            layers_vec_                                                   = layers_dict[system_name][component_name][sub_component_name]
            self.sub_components[sub_component_name]                       = Sub_Component(InputData, self.system_name, self.name, sub_component_name, layers_vec=layers_vec_, layer_names=layer_names)

    # ---------------------------------------------------------------------------------------------------------------------------
     

    # ---------------------------------------------------------------------------------------------------------------------------
    def call_classic(self, inputs, shift=None, training=False):

        return self.sub_components['Main'].call(inputs, training)

    # ---------------------------------------------------------------------------------------------------------------------------   


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_improved(self, inputs, shift=None, training=False):

        y = inputs

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



#=======================================================================================================================================
class TransLayer(tf.keras.layers.Layer):

    def __init__(self, f, NVars, indxs, name='TransLayer'):
        super(TransLayer, self).__init__(name=name, trainable=False)
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