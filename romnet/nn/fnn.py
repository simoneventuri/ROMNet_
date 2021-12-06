import numpy        as np
import tensorflow   as tf
import pandas       as pd

from .nn        import NN

class FNN(NN):
    """Fully-connected neural network.
    """

    #===================================================================================================================================
    def __init__(self, InputData, xnorm, NN_Transfer_Model):
        super(FNN, self).__init__()

        self.Name              = 'FNN'

        self.InputVars         = InputData.InputVars
        self.OutputVars        = InputData.OutputVars
        self.NVarsx            = len(InputData.InputVars)
        self.NVarsy            = len(InputData.OutputVars)
        
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

        self.TransFun          = InputData.TransFun



        self.FNNLayersVecs = {}
        for iFNN in range(self.NFNNs):

            ## FNN Block
            self.FNNLayersVecs[iFNN] = self.fnn_block(self.xnorm, '', 'NN', iFNN, self.InputVars)
        

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        OutputVec = []
        for iFNN in range(self.NFNNs):
            y  = inputs
    
            for f in self.FNNLayersVecs[iFNN]:
                y = f(y, training=training)

            OutputVec.append(y)

        if (self.NFNNs > 1):
            OutputConcat = tf.keras.layers.Concatenate(axis=1)(OutputVec)
        else:
            OutputConcat = OutputVec[0]

        if (self.AntiPCA_flg):
            OutputFinal = ROM.AntiPCALayer(A0=self.A_AntiPCA, C0=self.C_AntiPCA, D0=self.D_AntiPCA)(OutputConcat)
        else:
            OutputFinal = OutputConcat

        return OutputFinal

    #===================================================================================================================================



    #===================================================================================================================================
    def get_graph(self):
            input_  = tf.keras.Input(shape=[self.NVarsx,])
            return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================
