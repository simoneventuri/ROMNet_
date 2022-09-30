import numpy                                  as np
import tensorflow                             as tf
import h5py

from tensorflow.keras                     import regularizers
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras              import backend
from tensorflow.python.keras.engine       import base_layer_utils
from tensorflow.python.keras.utils        import tf_utils
from tensorflow.python.util.tf_export     import keras_export

from ..component                          import Component
from ..normalization                      import CustomNormalization

from .                                    import System_of_Components



#=======================================================================================================================================
class FNN_System(System_of_Components):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, name, norm_input, stat_input, layers_dict=[], layer_names_dict=[]):
        super(FNN_System, self).__init__()

        self.name                          = name

        if ('_' in self.name):
            self.type, self.idx            = self.name.split('_')
            self.idx                       = int(self.idx)
        else:           
            self.idx                       = 1
            self.type                      = self.name 
                  
        self.structure                     = InputData.structure

        self.input_vars                    = []
        for input_vars_ in InputData.input_vars[self.name].values():
            self.input_vars               += input_vars_
        self.input_vars                    = list(set(self.input_vars))
        self.n_inputs                      = len(self.input_vars)
        
        if (norm_input is not None):
            self.norm_input                = norm_input
        else:
            self.norm_input                = None
        if (stat_input is not None):
            self.stat_input                = stat_input
        else:
            self.stat_input                = None

        try:
            self.data_preproc_type         = InputData.data_preproc_type
        except:
            self.data_preproc_type         = 'std'

        self.output_vars                   = InputData.output_vars
        self.n_outputs                     = len(self.output_vars)

        try:
            self.internal_pca_flg          = InputData.internal_pca_flg
        except:
            self.internal_pca_flg          = False


        try:
            self.gaussnoise_rate           = InputData.gaussnoise_rate[self.name]
        except:
            self.gaussnoise_rate           = None

        try:
            self.softmax_flg               = InputData.softmax_flg[self.name]
        except:
            self.softmax_flg               = False
        if (self.softmax_flg):
            self.softmax_layer             = tf.keras.layers.Softmax(axis=1)

        try:
            self.rectify_flg               = InputData.rectify_flg
        except:
            self.rectify_flg               = False
        if (self.rectify_flg):
            self.rectify_layer             = tf.keras.activations.relu

        print("[ROMNet - system_of_components.py   ]:     Constructing System of Components: " + self.name) 


        # Normalizing Layer
        self.norm_layers_dict = []
        if (InputData.norm_input_flg[self.name]):

            for key, value in InputData.norm_input_flg[self.name].items():
                if (InputData.norm_input_flg[self.name][key]):

                    layer_name                              = self.name + '-' + key + '_Normalization'
                    layer                                   = CustomNormalization(name=layer_name, data_preproc_type=self.data_preproc_type)
                    norm_input                              = np.array(self.norm_input[InputData.input_vars[self.name][key]])
                    layer.adapt(norm_input)
                    if (not key in layers_dict[self.name]):
                        layers_dict[self.name][key]         = {}
                        layer_names_dict[self.name][key]    = {}
                    layers_dict[self.name][key][layer_name] = layer
                    layer_names_dict[self.name][key][layer_name] = layer_name

                    self.norm_layers_dict.append(key)


        # Gaussian Noise Layer
        self.noise_layers_dict = []
        if (self.gaussnoise_rate):

            for key, value in self.gaussnoise_rate.items():
                if (self.gaussnoise_rate[key]):

                    layer_name                              = self.name + '-' + key + '_GaussNoise'
                    layer                                   = tf.keras.layers.GaussianNoise(value)
                    if (not key in layers_dict[self.name]):
                        layers_dict[self.name][key]         = {}
                        ayer_names_dict[self.name][key]     = {}
                    layers_dict[self.name][key][layer_name] = layer
                    layer_names_dict[self.name][key][layer_name] = layer_name

                    self.noise_layers_dict.append(key)


        # Iterating over Components
        self.components     = {}
        for component_name in self.structure[self.name]:

            if ('FNN' in component_name.lower()):
                component_type = 'FNN'

            if (not component_name in layers_dict[self.name]):
                layers_dict[self.name][component_name]      = {}
                layer_names_dict[self.name][component_name] = {}

            self.components[component_name]            = Component(InputData, self.name, component_name, self.norm_input, layers_dict=layers_dict, layer_names_dict=layer_names_dict)



    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, layers_dict, training=False):

        if ('FNN' in self.norm_layers_dict):
            inputs = layers_dict[self.name]['FNN'][self.name+'-FNN_Normalization'](inputs)

        if ('FNN' in self.noise_layers_dict):
            inputs = layers_dict[self.name]['FNN'][self.name+'-FNN_GaussNoise'](inputs, training=training)


        y = self.components['FNN'].call(inputs, layers_dict, None, training=training)

        if (self.softmax_flg is True):
            # Apply SoftMax for Forcing Sum(y_i)=1

            output_T, output_species = tf.split(y, [1,self.n_outputs-1], axis=1)
            output_species           = self.softmax_layer(output_species)
            y                        = tf.keras.layers.Concatenate(axis=1)([output_T, output_species])
            # y                        = self.softmax_layer(y)

        if (self.rectify_flg):
            # Apply ReLu for Forcing y_i>0
            y                        = self.rectify_layer(y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================
