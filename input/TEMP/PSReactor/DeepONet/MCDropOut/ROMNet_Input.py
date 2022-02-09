import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNet_fld):

        #=======================================================================================================================================
        ### Case Name
        self.run_idx            = 0                                                                      # Training Case Identification Number 

        #=======================================================================================================================================
        ### Execution Flags
        self.DefineModelIntFlg   = 1
        self.train_int_flg         = 2                                                                      # Training                       0=>No, 1=>Yes
        self.WriteParamsIntFlg   = 1                                                                      # Writing Parameters             0=>Never, 1=>After Training, 2=>Also During Training
        self.WriteDataIntFlg     = 2                                                                      # Writing Data After Training    0=>Never, 1=>After Training, 2=>Also During Training
        self.TestIntFlg          = 2                                                                      # Evaluating                     0=>No, 1=>Yes
        self.plot_int_flg          = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training
        self.PredictIntFlg       = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                                                         # os.getenv('WORKSPACE_PATH')      
        self.ROMNet_fld          = ROMNet_fld                                                             # $WORKSPACE_PATH/ProPDE/
        self.path_to_run_fld        = self.ROMNet_fld   + '/../PSR_10Cases/'                                 # Path To Training Folder
        self.PathToTrainDataFld  = self.ROMNet_fld   + '/../Data/PSR_10Cases/pc_data_3/'                  # Path To Training Data Folder 
        self.PathToTestDataFld   = self.ROMNet_fld   + '/../Data/PSR_10Cases_Test/pc_data_3/'             # Path To Test Data Folder 

        #=======================================================================================================================================
        ### Data
        self.data_type            = 'BlackBox'                                                             # Module to Be Used for Reading Data

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type       = 'DeepONet'                                                             # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'MCDropOut'                                                            # Probabilistic Technique for Training the BNN (if Any)
        self.PINN                = False                                                                  # Flag for Training a Physics-Informed NN (in development)
        self.dOutputFile         = ''                                                                     # Name of the File Containing the ODE Residuals
        self.NormalizeInput      = True                                                                   # Flag for Normalizing Branch's Input Data
        self.InputFile           = 'Input.csv'                                                            # Name of the File Containing the Input Data
        self.BranchVars          = ['log10(Rest)']                                                        # List Containing the Branch's Input Data Column Names
        self.BranchScale         = None                                                                   # Function to Be Applied to the Input Data
        self.BranchLayers        = [np.array([32,32,32]), np.array([32,32,32]), np.array([32,32,32])]     # List Containing the No of Neurons per Each Branch's Layer
        self.BranchActFun        = [['elu','elu','elu'],  ['elu','elu','elu'],  ['elu','elu','elu']]      # List Containing the Activation Funct.s per Each Branch's Layer
        self.BranchDropOutRate   = 1.e-2                                                                  # Branch's Layers Dropout Rate
        self.BranchDropOutPredFlg= True                                                                   # Flag for Using Branch's Dropout during Prediction
        self.BranchSoftmaxFlg    = False                                                                  # Flag for Using Softmax after Branch's Last Layer
        self.TrunkVars           = ['log10(t)']                                                           # List Containing the Trunk's Input Data Column Names
        self.TrunkScale          = None                                                                   # Flag for Normalizing Trunk's Input Data
        self.TrunkLayers         = [np.array([32,32,32])]                                                 # List Containing the No of Neurons per Each Trunk's Layer
        self.TrunkActFun         = [['tanh','tanh','tanh']]                                               # List Containing the Activation Funct.s per Each Trunk's Layer
        self.TrunkDropOutRate    = 1.e-2                                                                  # Trunk's Layers Dropout Rate  
        self.TrunkDropOutPredFlg = True                                                                   # Flag for Using Trunk's Dropout during Prediction
        self.FinalLayerFlg       = True                                                                   # Flag for Using a Full Linear Layer after Dot-Product Layer
        self.OutputFile          = 'Output.csv'                                                           # Name of the File Containing the Output Data
        self.OutputVars          = ['PC_1','PC_2','PC_3']

        #=======================================================================================================================================
        ### Training Quanties
        self.transfer_flg         = False                                                                  # Flag for Using Transfer Learning
        self.TransferModelFld    = ''                                                                     # File Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch              = 50000                                                                  # Number of Epoches
        self.Minibatch_size       = 64                                                                     # Mini-Batch Size
        self.LossFunction        = 'mean_squared_error' #'mean_absolute_percentage_error'                 # Loss Function
        self.LearningRate        = 1.e-4                                                                  # Initial Learning Rate
        self.optimizer           = 'adam'                                                                 # Optimizer
        self.optimizer_params     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-10,1.e-4], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.ImpThold            = 1.e-6                                                                  # Importance Threshold for Early Stopping
        self.NPatience           = 300                                                                    # Patience Epoches for Early Stopping
        self.DecaySteps          = 30000                                                                  # No of Steps for Learning Rate Exponential Dacay
        self.DecayRate           = 0.98                                                                   # Rate for Learning Rate Exponential Dacay
        self.valid_perc           = 20.0                                                                   # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)

        #=======================================================================================================================================
        ### Testing Quantities
        self.TestPerc            = 0.0                                                                       # Percentage of Overall Data to Be Used for Testing (e.g., = 20.0 => 20%)
