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

        try:
            self.ULayers       = InputData.ULayers                          
            self.UActFun       = InputData.UActFun                          
            self.VLayers       = InputData.VLayers                          
            self.VActFun       = InputData.VActFun 
        except:
            self.ULayers       = None                
            self.UActFun       = None                
            self.VLayers       = None                
            self.VActFun       = None       


        try:
            self.SoftMaxFlg    = InputData.SoftMaxFlg
        except:
            self.SoftMaxFlg    = False

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

        self.PathToPODFile     = None


        self.FNNLayersVecs = {}
        self.FNNLayers_U   = {}
        self.FNNLayers_V   = {}
        for iFNN in range(self.NFNNs):

            ## FNN Block
            self.FNNLayersVecs[iFNN], self.FNNLayers_U[iFNN], self.FNNLayers_V[iFNN] = self.fnn_block(self.xnorm, '', 'NN', iFNN, self.InputVars)

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        OutputVec = []
        for iFNN in range(self.NFNNs):
            NLayers = len(self.FNNLayersVecs[iFNN])

            y = inputs
            if (self.NormalizeInput):
                iStart = 1
                y      = self.FNNLayersVecs[iFNN][0](y, training=training)
            else:
                iStart = 0 

            if (self.FNNLayers_U[iFNN]):
                y_U = self.FNNLayers_U[iFNN](y, training=training)
                y_V = self.FNNLayers_V[iFNN](y, training=training)

            for iLayer, f in enumerate(self.FNNLayersVecs[iFNN][iStart::]):
                y = f(y, training=training)
                if ( (self.FNNLayers_U[iFNN]) and (iLayer < NLayers-(1+iStart)) and (not 'dropout' in f.name) ):
                    yo = tf.keras.layers.Lambda(lambda x: 1.-x)(y)
                    ya = tf.keras.layers.multiply([yo, y_U])
                    yb = tf.keras.layers.multiply([   y, y_V])
                    y  = tf.keras.layers.add([ya, yb])

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
