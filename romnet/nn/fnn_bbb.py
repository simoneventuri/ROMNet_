import numpy                  as np
import tensorflow             as tf
import tensorflow_probability as tfp
import pandas                 as pd

from tensorflow_probability.python.distributions import kullback_leibler

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

        try:
            self.PathToLoadFile     = InputData.PathToLoadFile
        except:
            self.PathToLoadFile     = None

        self.BatchSize              = InputData.BatchSize
        try:
            self.SigmaLike          = InputData.SigmaLike
            self.CalibrateSigmaLFlg = False
        except:
            self.CalibrateSigmaLFlg = True


        self.PreLayersVecs = self.fnn_first_half_block_tfp(self.xnorm, '', 'NN', 0, self.InputVars)

        ## FNN Block
        self.FNNLayersVecs_Mu    = []
        self.FNNLayersVecs_Mu    = self.fnn_second_half_block_tfp('', 'NN', 0, '', LayersVec=self.FNNLayersVecs_Mu)
        self.FNNLayersVecs_Sigma = []
        self.FNNLayersVecs_Sigma = self.fnn_second_half_block_tfp('', 'NN', 1, 'Sigma', LayersVec=self.FNNLayersVecs_Sigma)

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        if (self.CalibrateSigmaLFlg):
            def normal_sp(OutputVec): 
                # params_1, params_2 = tf.split(OutputVec[0], 2, axis=1)
                # dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=OutputVec[0], scale_diag=(1e-8 + tf.math.softplus(0.05 * OutputVec[1])) )
                #dist = tfp.distributions.MultivariateNormalDiag(loc=OutputVec[0], scale_diag=(OutputVec[1]) )
                # dist = tfp.distributions.MultivariateNormalDiag(loc=OutputVec[0], scale_diag=(OutputVec[1]) )
                return dist
        else:
            def normal_sp(OutputVec): 
                params_1 = OutputVec[0]
                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=params_1, scale_diag=self.SigmaLike)
                return dist

        y0 = inputs
        for f in self.PreLayersVecs:
            y0 = f(y0, training=training)

        y = y0 
        for f in self.FNNLayersVecs_Mu:
            y = f(y, training=training)
        params_mu = y

        if (self.CalibrateSigmaLFlg):

            y = y0
            for f in self.FNNLayersVecs_Sigma:
                y = f(y, training=training)
            #params_sigma = tfp.bijectors.Softplus(hinge_softness=np.array([5.e-2, 5.e-2]), low=1.e-5, name='softplus')(y)
            params_sigma = y#tfp.bijectors.Exp(name='exp')(y) #tfp.bijectors.Softplus(hinge_softness=np.array([5.e-2, 5.e-2]), low=1.e-5, name='softplus')(y)

            OutputVec = [params_mu] + [params_sigma]

        else:

            OutputVec = [params_mu]

        OutputFinal = tfp.layers.DistributionLambda(normal_sp)(OutputVec) 


        return OutputFinal

    #===================================================================================================================================



    #===================================================================================================================================
    def get_graph(self):
        input_  = tf.keras.Input(shape=[self.NVarsx,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================
