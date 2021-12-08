import numpy        as np
import tensorflow   as tf
import pandas       as pd

from .nn        import NN
from ..pinn     import system as system_class

class AutoEncoder(NN):
    """Fully-connected neural network.
    """

    #===================================================================================================================================
    def __init__(self, InputData, xnorm, NN_Transfer_Model):
        super(AutoEncoder, self).__init__()

        self.Name              = 'AutoEncoder'

        if isinstance(InputData.InputVars, (list,tuple)):
            self.InputVars  = InputData.InputVars
        else:
            data_id         = list(InputData.InputVars.keys())[0]
            self.InputVars  = list(pd.read_csv(InputData.PathToDataFld+'/train/'+data_id+'/'+InputData.InputVars[data_id], header=None).to_numpy()[0,:])

        if isinstance(InputData.OutputVars, (list,tuple)):
            self.OutputVars = InputData.OutputVars
        else:
            data_id         = list(InputData.OutputVars.keys())[0]
            self.OutputVars = list(pd.read_csv(InputData.PathToDataFld+'/train/'+data_id+'/'+InputData.OutputVars[data_id], header=None).to_numpy()[0,:])

        self.NVarsx         = len(self.InputVars)
        self.NVarsy         = len(self.OutputVars)
        
        if (xnorm is None):
            xnorm = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=self.InputVars)

        self.xnorm            = xnorm
        self.NN_Transfer_Model = NN_Transfer_Model

        self.NormalizeInput    = InputData.NormalizeInput
        self.Layers            = InputData.Layers
        self.ActFun            = InputData.ActFun
        self.WeightDecay       = InputData.WeightDecay
        self.NFNNs             = len(InputData.Layers)

        self.DropOutRate       = InputData.DropOutRate
        self.DropOutPredFlg    = InputData.DropOutPredFlg

        if (np.sum(np.array(self.WeightDecay)) > 0.):
            self.RegularizeFlg = True
        else:
            self.RegularizeFlg = False

        self.attention_mask    = None

        self.ynorm_flg         = False

        try:
            self.TransFun      = InputData.TransFun
        except:
            self.TransFun      = None


        self.FNNLayers = []

        #self.FNNLayers.append( system_class.AutoEncoderLayer(InputData.PathToDataFld, self.NVarsx) )

        self.FNNLayers = self.fnn_block(self.xnorm, '', 'NN', 0, self.InputVars, LayersVec=self.FNNLayers)
        
        #self.FNNLayers.append( system_class.AntiAutoEncoderLayer(InputData.PathToDataFld, self.NVarsx) )

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        y  = inputs

        for f in self.FNNLayers:
            y = f(y, training=training)

        return y

    #===================================================================================================================================



    #===================================================================================================================================
    def get_graph(self):
        input_  = tf.keras.Input(shape=[self.NVarsx,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================
