import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, SurQCTFldr):

        #=======================================================================================================================================
        ### Case Name
        self.NNRunIdx            = 4                                                                      # Training Case Identification Number 

        #=======================================================================================================================================
        ### Execution Flags
        self.DefineModelIntFlg   = 1
        self.TrainIntFlg         = 2                                                                      # Training                       0=>No, 1=>Yes
        self.WriteParamsIntFlg   = 1                                                                      # Writing Parameters             0=>Never, 1=>After Training, 2=>Also During Training
        self.WriteDataIntFlg     = 2                                                                      # Writing Data After Training    0=>Never, 1=>After Training, 2=>Also During Training
        self.TestIntFlg          = 2                                                                      # Evaluating                     0=>No, 1=>Yes
        self.PlotIntFlg          = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training
        self.PredictIntFlg       = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                                                         # os.getenv('WORKSPACE_PATH')      
        self.SurQCTFldr          = SurQCTFldr                                                             # $WORKSPACE_PATH/ProPDE/
        self.PathToRunFld        = self.SurQCTFldr + '/../SMD_Test' + str(self.NNRunIdx)                   # Path To Training Folder
        self.TBCheckpointFldr    = self.PathToRunFld + '/TB/'
        self.PathToFigFld        = self.PathToRunFld + '/Figures/'                                        # Path To Training Figures Folder 
        self.PathToParamsFld     = self.PathToRunFld + '/Params/'                                         # Path To Training Parameters Folder 
        
        #=======================================================================================================================================
        ### Data
        self.DataType            = 'BlackBox'
        self.PathToDataFld       = self.WORKSPACE_PATH + '/ROMNet/Data_10SMD_Clean/'                   # Path To Training Data Folder 

        #=======================================================================================================================================
        ## NN Model Structure
        # self.ApproxModel         = 'DotNet'
        # self.BranchVars          = ['Rest']
        # self.BranchLayers        = [np.array([32,32,32])]
        # self.BranchActFun        = [['elu','elu','elu']]
        # self.BranchDropOutRate   = 1.e-10
        # self.TrunkVars           = ['t']
        # self.TrunkLayers         = [np.array([32,32,32])]
        # self.TrunkActFun         = [['tanh','tanh','tanh']]
        # self.TrunkDropOutRate    = 1.e-10
        # self.OutputVars          = ['PC1']

        self.ApproxModel         = 'DotNet'
        self.PINN                = False
        self.BranchVars          = ['x','v']
        self.BranchScale         = None
        self.BranchLayers        = [np.array([32,32,32]), np.array([32,32,32])]
        self.BranchActFun        = [['elu','elu','elu'],  ['elu','elu','elu']]
        self.BranchDropOutRate   = 1.e-2
        self.BranchDropOutTrain  = True
        self.TrunkVars           = ['t']
        self.TrunkScale          = None
        self.TrunkLayers         = [np.array([32,32,32])]
        self.TrunkActFun         = [['tanh','tanh','tanh']]
        self.TrunkDropOutRate    = 1.e-2
        self.TrunkDropOutTrain   = True
        self.OutputVars          = ['x','v']

        #=======================================================================================================================================
        ### Training Quanties
        self.NEpoch              = 300000                                                                       # Number of Epoches
        self.MiniBatchSize       = 32
        self.LossFunction        = 'mean_squared_error'#'mean_squared_error'#'mean_absolute_percentage_error'
        self.LearningRate        = 1.e-4                                                                     # Initial Learning Rate
        self.Optimizer           = 'adam'                                                                    # Optimizer Identificator
        self.OptimizerParams     = [0.9, 0.999, 1e-07]                                                       # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-10, 1.e-10], dtype=np.float64)                              # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.ImpThold            = 1.e-10  
        self.NPatience           = 1000 
        self.ValidPerc           = 20.0                                                                      # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.DecaySteps          = 30000
        self.DecayRate           = 0.98


        #=======================================================================================================================================
        ### Testing Quantities
        self.TestPerc            = 20.0                                                                       # Percentage of Overall Data to Be Used for Testing (e.g., = 20.0 => 20%)
