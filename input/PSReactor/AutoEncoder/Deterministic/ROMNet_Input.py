import os
import numpy      as np
import tensorflow as tf

#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNetFldr):

        self.NRODs               = 3

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
        self.PathToRunFld        = self.ROMNetFldr   + '/../PSR_100Cases/'                                 # Path To Training Folder
        self.PathToLoadFld       = None # self.ROMNetFldr   + '/../PSR_10Cases/DeepONet/Deterministic/Run_1/'                            # Path To Pre-Trained Model Folder
        self.PathToDataFld       = self.ROMNetFldr   + '/../Data/PSR_100Cases/Orig/'                        # Path To Training Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.PhysSystem          = 'PSR'                                                                  # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.DataType            = 'BlackBox'                                                             # Module to Be Used for Reading Data
        self.InputFiles          = {'ext':'Output.csv'}                                                   # Names of the Files Containing the Input Data
        self.OutputFiles         = {'ext':'Output.csv'}                                                  # Names of the Files Containing the Output Data        self.GenerateFlg         = False
        self.ValidPerc           = 20.0                                                                   # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.TestPerc            =  0.0                                                                   # Percentage of Overall Data to Be Used for Testing (e.g., = 20.0 => 20%)

       #=======================================================================================================================================
        ## NN Model Structure
        self.SurrogateType       = 'AutoEncoder'                                                          # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        self.InputVars           = {'ext':'CleanVars.csv'}
        self.OutputVars          = {'ext':'CleanVars.csv'}
        self.NormalizeInput      = False                                                                   # Flag for Normalizing Branch's Input Data
        # self.Layers              = [np.array([16,8,3,8,16])]                                           # List Containing the No of Neurons per Each NN's Layer
        # self.ActFun              = [['relu','relu','relu','relu','relu']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        self.Layers              = [np.array([16,16,16])]                                           # List Containing the No of Neurons per Each NN's Layer
        self.ActFun              = [['relu','relu','relu']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        self.DropOutRate         = 1.e-40                                                                  # NN's Layers Dropout Rate
        self.DropOutPredFlg      = False     
        self.NormalizeOutput     = False                                                                  # Flag for Normalizing Branch's Input Data

        #=======================================================================================================================================
        ### Training Quanties
        self.TransferFlg         = False                                                                  # Flag for Using Transfer Learning
        self.PathToTransFld      = ''                                                                     # Folder Containing the Trained Model to be Used for Transfer Learning 
        self.NEpoch              = 10000                                                                   # Number of Epoches
        self.BatchSize           = 32                                                                     # Mini-Batch Size
        self.ValidBatchSize      = 32                                                                     # Validation Mini-Batch Size
        self.RunEagerlyFlg       = False   
        self.Losses              = {'ext': {'name': 'mse', 'axis': 0}}                                    # Loss Functions
        self.LossWeights         = None  
        self.Metrics             = None                   
        self.LR                  = 5.e-6                                                               # Initial Learning Rate
        self.LRDecay             = ["exponential", 20000, 0.98]
        self.Optimizer           = 'adam'                                                                 # Optimizer
        self.OptimizerParams     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-18,1.e-18], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.Callbacks           = {
            'base': {
                'stateful_metrics': None
            },
            'early_stopping': {
                'monitor':              'val_loss',
                'min_delta':            1.e-8,
                'patience':             3000,
                'restore_best_weights': True,
                'verbose':              1,
                'mode':                 'auto',
                'baseline':             None
            },
            'model_ckpt': {
                'monitor':           'val_loss',
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
            #     'loss_weights0': {'ics': 1.e2, 'res': 1.e-1},
            #     'freq':          2
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
