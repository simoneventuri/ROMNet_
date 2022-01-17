import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNetFldr):

        self.iROD                = 1
        # POD_NAME                 = str(self.iROD)
        POD_NAME                 = 'All'

        self.NPODs               = 2
        self.NICs                = 100


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
        self.PathToRunFld        = self.ROMNetFldr   + '/../MSD_100Cases_POD_'+POD_NAME+'_Branch_'+str(self.iROD)+'/'           # Path To Training Folder
        self.PathToLoadFld       = None                                                                   # Path To Pre-Trained Model Folder
        # self.PathToDataFld       = self.ROMNetFldr   + '/../Data/MSD_100Cases/Orig/OneByOne/POD_'+POD_NAME+'/Branch/'           # Path To Training Data Folder 
        # self.PathToDataFld       = self.ROMNetFldr   + '/../Data/MSD_100Cases/Orig/OneByOne/POD_'+POD_NAME+'/Branch_ICs/'           # Path To Training Data Folder 
        self.PathToDataFld       = self.ROMNetFldr   + '/../Data/MSD_100Cases/Orig/All/POD_'+POD_NAME+'/Branch_'+str(self.iROD)+'/'           # Path To Training Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.PhysSystem          = 'MassSpringDamper'                                                        # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.DataType            = 'PDE'                                                                    # Module to Be Used for Reading Data
        self.GenerateFlg         = False
        self.NTrain              = {'pts': 64}                                                         # No of Training Cases
        self.ValidPerc           = 20.0                                                                     # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.DataDist            = 'uniform'                                                                # Distribution for Sampling Independent Variables
        self.NTest               = 2                                                                        # No of Test Cases
        self.TestFlg             = False

        # #=======================================================================================================================================
        # ## NN Model Structure
        # self.SurrogateType       = 'FNN'                                                                  # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        # self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        # self.InputVars           = ['PC_'+str(iPC+1) for iPC in range(self.NRODs)]                                                                 # List Containing the Input Data Column Names 
        # self.OutputVars          = ['POD_'+str(iPOD+1) for iPOD in range(self.NPODs)]                                                              # List Containing the Output Data Column Names
        # self.TransFun            = None#{'log': ['t']} 
        # self.NormalizeInput      = True                                                                   # Flag for Normalizing Input Data
        # self.Layers              = [np.array([32,64,256,500,self.NPODs])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.ActFun              = [['tanh','tanh','tanh','tanh','linear']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        # self.DropOutRate         = 1.e-4                                                                 # NN's Layers Dropout Rate
        # self.DropOutPredFlg      = False                                                                  # Flag for Using NN's Dropout during Prediction
        # self.NormalizeOutput     = True                                                                   # Flag for Normalizing Input Data


        #=======================================================================================================================================
        ## NN Model Structure
        self.SurrogateType       = 'FNN'                                                                  # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        self.InputVars           = ['x','v']                                                                 # List Containing the Input Data Column Names 
        self.OutputVars          = ['POD_'+str(iPOD+1) for iPOD in range(self.NPODs)]+['C']                                                              # List Containing the Output Data Column Names
        #self.OutputVars          = ['IC_'+str(iC+1) for iC in range(self.NICs)]                                                              # List Containing the Output Data Column Names
        self.TransFun            = None#{'log': ['t']} 
        self.NormalizeInput      = True                                                                   # Flag for Normalizing Input Data
        self.Layers              = [np.array([32,32,32,self.NPODs+1])]                                           # List Containing the No of Neurons per Each NN's Layer
        #self.Layers              = [np.array([32,32,32,self.NICs])]                                           # List Containing the No of Neurons per Each NN's Layer
        self.ActFun              = [['sigmoid','sigmoid','sigmoid','linear']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        #self.ActFun              = [['sigmoid','sigmoid','sigmoid','linear']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        self.DropOutRate         = 1.e-3                                                               # NN's Layers Dropout Rate
        self.DropOutPredFlg      = False                                                                  # Flag for Using NN's Dropout during Prediction
        self.NormalizeOutput     = False
        self.SoftMaxFlg          = False      

        # self.ULayers             = [np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.UActFun             = [['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        # self.VLayers             = [np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.VActFun             = [['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer

        #=======================================================================================================================================
        ### Training Quanties
        self.TransferFlg         = False                                                                  # Flag for Using Transfer Learning
        self.PathToTransFld      = ''                                                                     # Folder Containing the Trained Model to be Used for Transfer Learning 
        self.NEpoch              = 100000                                                                 # Number of Epoches
        self.BatchSize           = 32                                                                    # Batch Size for Training
        self.ValidBatchSize      = 32                                                                    # Batch Size for Validation
        self.Losses              = {'pts': {'name': 'MSE', 'axis': 0}} # Loss Functions
        #self.Losses              = {'pts': {'name': 'CCE', 'axis': 0}} # Loss Functions
        self.LossWeights         = {'pts': 1.}  
        self.Metrics             = None                                                                   # List of Metric Functions
        self.LR                  = 1.e-3                                                                # Initial Learning Rate
        self.LRDecay             = ["exponential", 500, 0.95]
        self.Optimizer           = 'adam'                                                                 # Optimizer
        self.OptimizerParams     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-10, 1.e-10], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.Callbacks           = {
            'base': {
                'stateful_metrics': None
            },
            'early_stopping': {
                'monitor':              'val_tot_loss',
                'min_delta':            1.e-8,
                'patience':             5000,
                'restore_best_weights': True,
                'verbose':              1,
                'mode':                 'auto',
                'baseline':             None
            },
            'model_ckpt': {
                'monitor':           'val_tot_loss',
                'save_best_only':    True,
                'save_weights_only': True,
                'verbose':           0, 
                'mode':              'auto', 
                'save_freq':         'epoch', 
                'options':           None
            },
            # 'tensorboard': {
            #     'histogram_freq':         0,
            #     'write_graph':            True,
            #     'write_grads':            True,
            #     'write_images':           True,
            #     'profile_batch':          0,
            #     'embeddings_freq':        0, 
            #     'embeddings_layer_names': None, 
            #     'embeddings_metadata':    None, 
            #     'embeddings_data':        None
            # },
            'lr_tracker': {
                'verbose': 1
            },
            # 'weighted_loss': {
            #     'name':          'SoftAttention',
            #     'data_generator': None,
            #     'loss_weights0': {'ics': 1., 'res': 1e-10},
            #     'freq':          5
            # },
            # 'weighted_loss': {
            #     'name':         'EmpiricalWeightsAdapter',
            #     'alpha':        0.9,
            #     'freq':         50,
            #     'max_samples':  20000
            # }
            # 'plotter': {
            #     **inp_post.plot_style,
            #     'freq': 200
            # },
        }
