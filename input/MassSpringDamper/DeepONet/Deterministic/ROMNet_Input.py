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
        self.PathToRunFld        = self.ROMNetFldr   + '/../MSD_100Cases_All/'                                 # Path To Training Folder
        # self.PathToLoadFld       = '/Users/sventuri/WORKSPACE/ROMNet/Data/MSD_100Cases/Orig/OneByOne/FNN/Final.h5'           # Path To Pre-Trained Model Folder
        self.PathToLoadFld       = None#'/Users/sventuri/WORKSPACE/ROMNet/Data/MSD_100Cases/Orig/All/FNN/Final.h5'             # Path To Pre-Trained Model Folder
        self.PathToDataFld       = self.ROMNetFldr   + '/../Data/MSD_100Cases/Orig/'                            # Path To Training Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.PhysSystem          = 'MassSpringDamper'                                                     # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.DataType            = 'PDE'                                                                    # Module to Be Used for Reading Data
        self.GenerateFlg         = False
        # self.NTrain              = {'ics': 64, 'res': 128}                                                  # No of Training Cases
        self.NTrain              = {'pts': 128}                                                  # No of Training Cases
        self.ValidPerc           = 20.0                                                                     # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.DataDist            = 'uniform'                                                                # Distribution for Sampling Independent Variables
        self.NTest               = 2                                                                        # No of Test Cases
        self.TestFlg             = False

        #=======================================================================================================================================
        ## NN Model Structure
        self.SurrogateType       = 'DeepONet'                                                             # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        self.NormalizeInput      = True                                                                   # Flag for Normalizing Branch's Input Data
        self.BranchToTrunk       = [0,0]                                                                  # Index of the Trunk Corresponding to i-th Branch
        self.BranchVars          = ['x','v']                                                              # List Containing the Branch's Input Data Column Names
        self.BranchLayers        = [np.array([32,32,32,4]), np.array([32,32,32,4])]                           # List Containing the No of Neurons per Each Branch's Layer
        self.BranchActFun        = [['sigmoid','sigmoid','sigmoid','linear']]*2                      # List Containing the Activation Funct.s per Each Branch's Layer
        # self.BranchLayers        = [np.array([32,32,32,100]), np.array([32,32,32,100])]                           # List Containing the No of Neurons per Each Branch's Layer
        # self.BranchActFun        = [['relu','relu','relu','linear']]*2                      # List Containing the Activation Funct.s per Each Branch's Layer
        self.BranchDropOutRate   = 1.e-10                                                                 # Branch's Layers Dropout Rate
        self.BranchDropOutPredFlg= False                                                                  # Flag for Using Branch's Dropout during Prediction
        # self.BranchSoftmaxFlg    = False                                                                  # Flag for Using Softmax after Branch's Last Layer
        self.BranchSoftmaxFlg    = False                                                                  # Flag for Using Softmax after Branch's Last Layer
        self.TrunkVars           = ['t']                                                                  # List Containing the Trunk's Input Data Column Names
        self.TrunkLayers         = [np.array([32,32,32,2])]                                                 # List Containing the No of Neurons per Each Trunk's Layer
        self.TrunkActFun         = [['sigmoid','sigmoid','sigmoid','linear']]#,
                                    #['tanh','tanh','tanh','linear']]                                               # List Containing the Activation Funct.s per Each Trunk's Layer
        self.TrunkDropOutRate    = 1.e-10                                                                  # Trunk's Layers Dropout Rate  
        self.TrunkDropOutPredFlg = False                                                                  # Flag for Using Trunk's Dropout during Prediction
        self.TransFun            = None
        self.FinalLayerFlg       = None                                                                   # Flag for Using a Full Linear Layer after Dot-Product Layer
        self.OutputVars          = ['x','v']                                                              # List Containing the Output Data Column Names

        # self.BranchULayers       = [np.array([32]),np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.BranchUActFun       = [['tanh'],['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        # self.BranchVLayers       = [np.array([32]),np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.BranchVActFun       = [['tanh'],['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer

        # self.TrunkULayers        = [np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.TrunkUActFun        = [['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        # self.TrunkVLayers        = [np.array([32])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.TrunkVActFun        = [['tanh']]                                 # List Containing the Activation Funct.s per Each NN's Layer

        #=======================================================================================================================================
        ### Training Quanties
        self.TrainBranchFlg      = True
        self.TrainTrunkFlg       = True
        self.TransferFlg         = False                                                                  # Flag for Using Transfer Learning
        self.PathToTransFld      = ''                                                                     # Folder Containing the Trained Model to be Used for Transfer Learning 
        self.NEpoch              = 10000                                                                  # Number of Epoches
        self.BatchSize           = 32                                                                    # Mini-Batch Size
        self.ValidBatchSize      = 32                                                                    # Validation Mini-Batch Size
        # self.Losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}} # Loss Functions
        # self.LossWeights         = {'ics': 1., 'res': 1.}     
        self.Losses              = {'pts': {'name': 'MSE', 'axis': 0}} # Loss Functions
        self.LossWeights         = {'pts': 1.}     
        self.Metrics             = None                                                                   # List of Metric Functions
        self.LR                  = 1.e-4                                                               # Initial Learning Rate
        self.LRDecay             = ["exponential", 10000, 0.98]
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
                'patience':             3000,
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
            #     'loss_weights0': {'ics': 1., 'res': 1.e-10},
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
