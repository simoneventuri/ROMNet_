import numpy        as np
import tensorflow   as tf
import pandas       as pd

from .nn        import NN

class DeepONet(NN):
    """Deep operator network 
    """

    #===================================================================================================================================
    def __init__(self, InputData, xnorm, NN_Transfer_Model):
        super(DeepONet, self).__init__()

        self.Name                 = 'DeepONet'

        self.BranchVars           = InputData.BranchVars
        self.TrunkVars            = InputData.TrunkVars
        self.NVarsx               = len(InputData.BranchVars + InputData.TrunkVars)

        if (xnorm is None):
            xnorm = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=self.BranchVars+self.TrunkVars)

        self.xnormBranch          = xnorm[InputData.BranchVars]
        self.xnormTrunk           = xnorm[InputData.TrunkVars] 
        self.NN_Transfer_Model    = NN_Transfer_Model

        self.OutputVars           = InputData.OutputVars
        self.NVarsy               = len(InputData.OutputVars)

        self.NormalizeInput       = InputData.NormalizeInput
        self.WeightDecay          = InputData.WeightDecay

        self.NTrunks              = len(InputData.TrunkLayers)
        self.NVarsTrunk           = len(InputData.TrunkVars)
        self.TrunkLayers          = InputData.TrunkLayers
        self.TrunkActFun          = InputData.TrunkActFun
        self.TrunkDropOutRate     = InputData.TrunkDropOutRate
        self.TrunkDropOutPredFlg  = InputData.TrunkDropOutPredFlg

        self.NBranches            = len(InputData.BranchLayers)
        self.NVarsBranch          = len(InputData.BranchVars)
        self.BranchLayers         = InputData.BranchLayers
        self.BranchActFun         = InputData.BranchActFun
        self.BranchDropOutRate    = InputData.BranchDropOutRate
        self.BranchDropOutPredFlg = InputData.BranchDropOutPredFlg
        self.BranchSoftmaxFlg     = InputData.BranchSoftmaxFlg
        self.BranchToTrunk        = InputData.BranchToTrunk

        self.FinalLayerFlg        = InputData.FinalLayerFlg

        self.TransFun             = InputData.TransFun

        if (np.sum(np.array(self.WeightDecay)) > 0.):
            self.RegularizeFlg    = True
        else:   
            self.RegularizeFlg    = False
   
        self.attention_mask       = None



        ### Trunks
        self.TrunkLayersVecs = {}
        for iTrunk in range(self.NTrunks):
            self.TrunkLayersVecs[iTrunk] = self.fnn_block(self.xnormTrunk, 'Trunk', 'Trunk_'+str(iTrunk+1), iTrunk)


        self.BranchLayersVecs = {}
        self.FinalLayersVecs  = {}
        for iy in range(self.NVarsy):

            ### Branches
            self.BranchLayersVecs[iy] = self.fnn_block(self.xnormBranch, 'Branch', 'Branch_'+InputData.OutputVars[iy], iy)
       
            ### Final Layer
            self.FinalLayersVecs[iy]  = self.deeponet_final_layer(iy)

    #===================================================================================================================================



    #===================================================================================================================================
    def call(self, inputs, training=False):

        inputsBranch, inputsTrunk = tf.split(inputs, num_or_size_splits=[len(self.BranchVars), len(self.TrunkVars)], axis=1)
    

        TrunkVec = []
        for iTrunk in range(self.NTrunks):
            y = inputsTrunk
            
            for f in self.TrunkLayersVecs[iTrunk]:
                y = f(y, training=training)

            TrunkVec.append(y)


        OutputVec = []        
        for iy in range(self.NVarsy):
            iTrunk = self.BranchToTrunk[iy]
            y      = inputsBranch

            for f in self.BranchLayersVecs[iy]:
                y = f(y, training=training)
            
            OutputP = tf.keras.layers.Dot(axes=1)([y, TrunkVec[iTrunk]])

            OutputVec.append( self.FinalLayersVecs[iy](OutputP, training=training) )


        if (self.NVarsy > 1):
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
            input_     = tf.keras.Input(shape=[self.NVarsx,])
            return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    #===================================================================================================================================
