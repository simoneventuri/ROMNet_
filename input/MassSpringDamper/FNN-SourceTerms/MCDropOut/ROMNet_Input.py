import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNetFldr):

        #=======================================================================================================================================
        ### Case Name
        self.NNRunIdx            = 0                                                                      # Training Case Identification Number 

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
        self.ROMNetFldr          = ROMNetFldr                                                             # $WORKSPACE_PATH/ProPDE/
        self.PathToRunFld        = self.ROMNetFldr   + '/../MSD_10Cases/'                                 # Path To Training Folder
        self.PathToTrainDataFld  = self.ROMNetFldr   + '/../Data/MSD_10Cases/'                            # Path To Training Data Folder 
        self.PathToTestDataFld   = self.ROMNetFldr   + '/../Data/MSD_10Cases_Test/'                       # Path To Test Data Folder 

        #=======================================================================================================================================
        ### Data
        self.DataType            = 'BlackBox'                                                             # Module to Be Used for Reading Data

        #=======================================================================================================================================
        ## NN Model Structure
        self.SurrogateType       = 'FNN-SourceTerms'                                                      # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'MCDropOut'                                                            # Probabilistic Technique for Training the BNN (if Any)
        self.PINN                = False                                                                  # Flag for Training a Physics-Informed NN (in development)
        self.dOutputFile         = ''                                                                     # Name of the File Containing the ODE Residuals
        self.NormalizeInput      = True                                                                   # Flag for Normalizing Input Data
        self.InputFile           = 'Output.csv'                                                           # Name of the File Containing the Input Data
        self.InputVars           = ['x','v']                                                              # List Containing the Input Data Column Names 
        self.InputScale          = None                                                                   # Function to Be Applied to the Input Data
        self.Layers              = [np.array([16,32,64,32,16])]                                           # List Containing the No of Neurons per Each NN's Layer
        self.ActFun              = [['tanh','tanh','tanh','tanh','tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        self.DropOutRate         = 1.e-2                                                                  # NN's Layers Dropout Rate
        self.DropOutPredFlg      = True                                                                   # Flag for Using NN's Dropout during Prediction
        self.OutputFile          = 'dOutput.csv'                                                          # Name of the File Containing the Output Data
        self.OutputVars          = ['dx','dv']                                                            # List Containing the Output Data Column Names

        #=======================================================================================================================================
        ### Training Quanties
        self.TransferFlg         = False                                                                  # Flag for Using Transfer Learning
        self.TransferModelFld    = ''                                                                     # File Containing the Trained Model to be Used for Transfer Learning 
        self.NEpoch              = 100000                                                                 # Number of Epoches
        self.MiniBatchSize       = 64                                                                     # Mini-Batch Size
        self.LossFunction        = 'mean_squared_error' #'mean_absolute_percentage_error'                 # Loss Function
        self.LearningRate        = 5.e-4                                                                  # Initial Learning Rate
        self.Optimizer           = 'adam'                                                                 # Optimizer
        self.OptimizerParams     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-5, 1.e-5], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.ImpThold            = 1.e-10                                                                 # Importance Threshold for Early Stopping
        self.NPatience           = 3000                                                                   # Patience Epoches for Early Stopping
        self.DecaySteps          = 30000                                                                  # No of Steps for Learning Rate Exponential Dacay
        self.DecayRate           = 0.98                                                                   # Rate for Learning Rate Exponential Dacay
        self.ValidPerc           = 20.0                                                                   # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)


        #=======================================================================================================================================
        ### Testing Quantities
        self.TestPerc            = 20.0                                                                   # Percentage of Overall Data to Be Used for Testing (e.g., = 20.0 => 20%)
