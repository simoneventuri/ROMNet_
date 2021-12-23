import numpy                  as np
import tensorflow             as tf
import tensorflow_probability as tfp
import pandas                 as pd

from .nn                  import NN

class FNN_BbB(NN):
    """Fully-connected neural network.
    """

    #===================================================================================================================================
    def __init__(self, InputData, xnorm, NN_Transfer_Model):
        super(FNN_BbB, self).__init__()

        self.Name                   = 'FNN_BbB'

        if isinstance(InputData.InputVars, (list,tuple)):
            self.InputVars          = InputData.InputVars
        else:         
            data_id                 = list(InputData.InputVars.keys())[0]
            self.InputVars          = list(pd.read_csv(InputData.PathToDataFld+'/train/'+data_id+'/'+InputData.InputVars[data_id], header=None).to_numpy()[0,:])

        if isinstance(InputData.OutputVars, (list,tuple)):
            self.OutputVars         = InputData.OutputVars
        else:        
            data_id                 = list(InputData.OutputVars.keys())[0]
            self.OutputVars         = list(pd.read_csv(InputData.PathToDataFld+'/train/'+data_id+'/'+InputData.OutputVars[data_id], header=None).to_numpy()[0,:])
        
        self.NVarsx                 = len(self.InputVars)
        self.NVarsy                 = len(self.OutputVars)
                
        if (xnorm is None):
            xnorm = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=self.InputVars)

        self.xnorm                  = xnorm
        self.NN_Transfer_Model      = NN_Transfer_Model
     
        self.NormalizeInput         = InputData.NormalizeInput
        self.Layers                 = InputData.Layers
        self.ActFun                 = InputData.ActFun
        self.WeightDecay            = InputData.WeightDecay
        self.NFNNs                  = len(InputData.Layers)
     
        self.DropOutRate            = InputData.DropOutRate
        self.DropOutPredFlg         = InputData.DropOutPredFlg
     
        if (np.sum(np.array(self.WeightDecay)) > 0.):
            self.RegularizeFlg      = True
        else:
            self.RegularizeFlg      = False

        self.attention_mask         = None

        self.ynorm_flg              = False

        try:
            self.TransFun           = InputData.TransFun
        except:
            self.TransFun           = None

        self.PathToPODFile          = None


        self.BatchSize              = InputData.BatchSize
        try:
            self.SigmaLike          = InputData.SigmaLike
            self.CalibrateSigmaLFlg = False
        except:
            self.CalibrateSigmaLFlg = True




        # self.FNNLayersVecs = {}
        # for iFNN in range(self.NFNNs):

        #     ## FNN Block
        #     self.FNNLayersVecs[iFNN] = self.fnn_block_tfp(self.xnorm, '', 'NN', iFNN, self.InputVars)



        self.PreLayersVecs = self.fnn_first_half_block_tfp(self.xnorm, '', 'NN', 0, self.InputVars)

        self.FNNLayersVecs = {}
        for iFNN in range(self.NFNNs):

            ## FNN Block
            self.FNNLayersVecs[iFNN] = self.fnn_second_half_block_tfp('', 'NN', iFNN)


    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        y0 = inputs
        for f in self.PreLayersVecs:
            y0 = f(y0, training=training)

        OutputVec = []
        for iFNN in range(self.NFNNs):
            y = y0

            for f in self.FNNLayersVecs[iFNN]:
                y = f(y, training=training)

            OutputVec.append(y)



        # OutputVec = []
        # for iFNN in range(self.NFNNs):
        #     y  = inputs
    
        #     for f in self.FNNLayersVecs[iFNN]:
        #         y = f(y, training=training)

        #     OutputVec.append(y)



        if (self.CalibrateSigmaLFlg):

            def normal_sp(OutputVec): 
                # params_1 = OutputVec[0]
                # params_2 = OutputVec[1]
                params_1, params_2 = tf.split(OutputVec[0], 2, axis=1)
                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=params_1, scale_diag=(1e-8 + tf.math.softplus(0.05 * params_2)) )
                return dist
        else:

            def normal_sp(OutputVec): 
                params_1 = OutputVec[0]
                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=params_1, scale_diag=self.SigmaLike)
                return dist

        OutputFinal = tfp.layers.DistributionLambda(normal_sp)(OutputVec) 


        return OutputFinal

    #===================================================================================================================================



    #===================================================================================================================================
    def get_graph(self):
            input_  = tf.keras.Input(shape=[self.NVarsx,])
            return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================
