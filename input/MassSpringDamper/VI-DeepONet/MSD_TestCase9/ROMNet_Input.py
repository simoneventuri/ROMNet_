import os
import numpy as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH):

        self.n_modes              = 2                                                                        # No of Modes (i.e., No of Neurons in Trunk's Last Layer)

        #=======================================================================================================================================
        ### Case Name
        self.run_idx              = 0                                                                        # Training Case Identification Number. If ==0, the code automatically assigns one.

        #=======================================================================================================================================
        ### Execution Flags
        self.train_int_flg        = 1                                                                        # Integer Flag for Training (0=>No, 1=>Yes)

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                                                            # os.getenv('WORKSPACE_PATH')       
        self.ROMNet_fld          = self.WORKSPACE_PATH + '/ROMNet/romnet/'                                   # $WORKSPACE_PATH/ROMNet/romnet/
        self.path_to_run_fld     = self.ROMNet_fld + '/../MSD_50Cases_wNoise_All_5em2t/'                                 # Path To Training Folder
        self.path_to_load_fld    = None #self.ROMNet_fld + '/../Data/MSD_100Cases/Orig/OneByOne/FNN/Final.h5'  # Path To Pre-Trained Model Folder 
        # self.path_to_load_fld    = self.ROMNet_fld +'/../MSD_100Cases_All/DeepONet/Deterministic/Run_2/'     # Path To Pre-Trained Model Folder 
        self.path_to_data_fld    = self.ROMNet_fld + '/../Data/MSD_50Cases_wNoise_5em2t/Orig/'                           # Path To Training-Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.phys_system         = 'MassSpringDamper'                                                        # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.data_type           = 'PDE'                                                                     # Module to Be Used for Reading Data
        self.generate_flg        = False                                                                     # Flag for Generating Data
        ## Fully Data Driven
        self.n_train             = {'pts': 0}                                                                # Type/No of Data Points
        ### Physics Informed
        # self.n_train             = {'ics': 0, 'res': 0}                                                      # Type/No of Data Points
        # self.valid_perc          = 20.0                                                                      # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        # self.data_dist           = 'uniform'                                                                 # Distribution for Sampling Independent Variables
        # self.test_flg             = False                                                                    # Test Flag
        # self.n_test               = 2                                                                        # No of Test Cases

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type      = 'VI_DeepONet'                                                             # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.plot_graph_flg      = False                                                                      # Flag for Plotting and Saving the Graph for the Network Structure
        self.trans_fun           = None #{'log': ['t']}                                                      # Dictionary Containing Functions to Be Applied to Input Data 
        self.norm_output_flg     = True                                                                     # Flag for Normalizing Output Data
        self.output_vars         = ['x','v']                                                                 # List Containing the Output Data Variable Names for each System
        self.input_vars_all      = ['x','v','t']                                                             # List Containing all the Input Data Variable Names
        self.input_vars          = {'DeepONetMean': {'Branch': ['x','v'],  
                                                      'Trunk': ['t']},
                                    'DeepONetSD':   {'Branch': ['x','v'],  
                                                      'Trunk': ['t']}}                                      # Dictionary Containing the Input  Data Variable Names for each Component
        self.norm_input_flg      = {'DeepONetMean': {'Branch_1': True,   
                                                     'Branch_2': True,   
                                                      'Trunk_1': True,
                                                      'Trunk_2': True},
                                    'DeepONetSD':   {'Branch_1': True,   
                                                     'Branch_2': True,   
                                                      'Trunk_1': True,
                                                      'Trunk_2': True}}                                     # Dictionary Containing Flags for Normalizing Input Data for each Component
        self.structure           = {'DeepONetMean': {'Branch_1': ['Main'],  
                                                     'Branch_2': ['Main'],  
                                                      'Trunk_1': ['Main'],
                                                      'Trunk_2': ['Main']},
                                     'DeepONetSD':  {'Branch_1': ['Main'],  
                                                     'Branch_2': ['Main'],  
                                                      'Trunk_1': ['Main'],
                                                      'Trunk_2': ['Main']}}                                 # Dictionary Containing the Structure of the Network
        self.branch_to_trunk     = {'DeepONetMean': 'one_to_one',
                                    'DeepONetSD':   'one_to_one'}                                           # DeepONet Branch-to-Trunk Type of Mapping  ('one_to_one'/'multi_to_one')
        self.n_branch_out        = self.n_modes                                                              # No of Neurons in Branches' Last Layers
        self.n_trunk_out         = self.n_modes                                                              # No of Neurons in Trunks' Last Layers
        self.layer_type          = {'DeepONetMean': {'Branch': {'Main': ['TFP','TFP','TFP','TFP','TFP']},  
                                                       'Trunk': {'Main': ['TFP','TFP','TFP','TFP','TFP']}},
                                    'DeepONetSD':   {'Branch': {'Main': ['TF','TF','TF','TF']},  
                                                       'Trunk': {'Main': ['TF','TF','TF','TF']}}}       # Dictionary Containing the Activation Funct.s for each Layer
        self.n_neurons           = {'DeepONetMean': {'Branch': {'Main': np.array([32,32,32,32,self.n_branch_out])},  
                                                       'Trunk': {'Main': np.array([32,32,32,32,self.n_trunk_out])}},
                                    'DeepONetSD':   {'Branch': {'Main': np.array([32,32,32,self.n_branch_out])},  
                                                       'Trunk': {'Main': np.array([32,32,32,self.n_trunk_out])}}} # Dictionary Containing the No of Neurons for each Layer
        self.act_funcs           = {'DeepONetMean': {'Branch': {'Main': ['tanh','tanh','tanh','tanh','linear']},  
                                                       'Trunk': {'Main': ['tanh','tanh','tanh','tanh','linear']}},
                                    'DeepONetSD':   {'Branch': {'Main': ['tanh','tanh','tanh','sigmoid']},  
                                                       'Trunk': {'Main': ['tanh','tanh','tanh','sigmoid']}}}       # Dictionary Containing the Activation Funct.s for each Layer
        self.dropout_rate        = {'DeepONetMean': {'Branch': {'Main': None},  
                                                       'Trunk': {'Main': None}},
                                    'DeepONetSD':   {'Branch': {'Main': None},  
                                                       'Trunk': {'Main': None}}}                                # Dictionary Containing the Dropout Rate for each Sub-Component
        self.dropout_pred_flg    = {'DeepONetMean': {'Branch': {'Main': False},  
                                                       'Trunk': {'Main': False}},
                                    'DeepONetSD':   {'Branch': {'Main': False},  
                                                       'Trunk': {'Main': False}}}                                 # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        self.softmax_flg         = {'DeepONetMean': {'Branch': {'Main': False},  
                                                       'Trunk': {'Main': False}},
                                    'DeepONetSD':   {'Branch': {'Main': False},  
                                                       'Trunk': {'Main': False}}}                                 # Dictionary Containing the Softmax Flag for each Sub-Component 
        self.dotlayer_bias_flg   = {'DeepONetMean': True,
                                      'DeepONetSD': True}
        self.sigma_like          = None

        #=======================================================================================================================================
        ### Losses
        ## Fully Data Driven
        self.losses              = {'pts': {'name': 'NLL', 'axis': 0}}                                       # Dictionary Containing Loss Functions for Each Data Type
        self.loss_weights        = {'pts': 1.}                                                               # Dictionary Containing Weights for Each Data Type
        # ## Physics Informed
        # self.losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}}    # Dictionary Containing Loss Functions for Each Data Type
        # self.loss_weights        = {'ics': 1., 'res': 1.}                                                    # Dictionary Containing Weights for Each Data Type
        self.metrics             = None                                                                      # List of Metrics                                                          
        self.run_eagerly_flg     = False

        #=======================================================================================================================================
        ### Training Quanties
        self.trainable_flg       = {'DeepONetMean': 'all',
                                    'DeepONetSD':   'all'}                                                       # Dictionary Containing Instructions for Training Components ('all'/'none'/'only_last')
        self.transfer_flg        = False                                                                     # Flag for Transfer Learning
        self.path_to_transf_fld  = ''                                                                        # Path to Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch             = 100000                                                                     # Number of Epoches
        self.batch_size          = 256                                                                        # Mini-Batch Size
        self.valid_batch_size    = 64                                                                       # Validation Mini-Batch Size
        self.lr                  = 1.e-3                                                                     # Initial Learning Rate
        self.lr_decay            = ["exponential", 1000, 0.90]                                              # Instructions for Learning Rate Decay
        self.optimizer           = 'adam'                                                                    # Optimizer
        self.optimizer_params    = [0.9, 0.999, 1e-07]                                                       # Parameters for the Optimizer
        self.weight_decay_coeffs = np.array([1.e-16, 1.e-16], dtype=np.float64)                              # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.callbacks_dict           = {
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
