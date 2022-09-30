import os
import numpy as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH):

        self.n_modes              = 1                                                                        # No of Modes (i.e., No of Neurons in Trunk's Last Layer)

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
        self.path_to_run_fld     = self.ROMNet_fld + '/../Rect_100Instants_TransRotScale/'                              # Path To Training Folder
        self.path_to_load_fld    = None #self.ROMNet_fld + '/../Data/MSD_100Cases/Orig/OneByOne/FNN/Final.h5'  # Path To Pre-Trained Model Folder 
        # self.path_to_load_fld    = self.ROMNet_fld +'/../MSD_100Cases_All/DeepONet/Deterministic/Run_2/'     # Path To Pre-Trained Model Folder 
        self.path_to_data_fld    = self.ROMNet_fld + '/../Data/Rect_100Instants_TransRotScale/Orig/'                           # Path To Training-Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.phys_system         = 'Generic'                                                            # Name of the Physical System for PINN

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
        self.surrogate_type      = 'DeepONet'                                                                # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.plot_graph_flg      = True                                                                      # Flag for Plotting and Saving the Graph for the Network Structure
        self.trans_fun           = None #{'log': ['t']}                                                      # Dictionary Containing Functions to Be Applied to Input Data 
        self.norm_output_flg     = False                                                                     # Flag for Normalizing Output Data
        #self.data_preproc_type   = 'range'
        self.output_vars         = ['z']                                                                 # List Containing the Output Data Variable Names for each System
        self.input_vars_all      = ['t','x','y']                                                             # List Containing all the Input Data Variable Names
        self.input_vars          = {'DeepONet': {'Branch': ['t'],  
                                                  'Shift': ['t'],
                                                'Stretch': ['t'],
                                               'Rotation': ['t'],
                                                  'Trunk': ['x','y']}}                                          # Dictionary Containing the Input  Data Variable Names for each Component
        self.norm_input_flg      = {'DeepONet': {'Branch': False,   
                                                  'Shift': False,
                                                'Stretch': False,
                                               'Rotation': False,
                                                  'Trunk': False}}
        self.structure           = {'DeepONet': {'Branch': ['Main'],  
                                                  'Shift': ['Main'],  
                                                'Stretch': ['Main'],  
                                               'Rotation': ['Main'],  
                                                  'Trunk': ['Main']}}                                      # Dictionary Containing the Structure of the Network
        self.branch_to_trunk     = {'DeepONet': 'one_to_one'}                                                # DeepONet Branch-to-Trunk Type of Mapping  ('one_to_one'/'multi_to_one')
        self.n_branch_out        = self.n_modes+1                                                              # No of Neurons in Branches' Last Layers
        self.n_trunk_out         = self.n_modes                                                              # No of Neurons in Trunks' Last Layers
        self.n_neurons           = {'DeepONet': {'Branch': {'Main': np.array([8,8,self.n_branch_out])},
                                                  'Shift': {'Main': np.array([16,16,16,2])},  
                                                'Stretch': {'Main': np.array([16,16,16,1])},  
                                               'Rotation': {'Main': np.array([16,16,16,1])},  
                                                  'Trunk': {'Main': np.array([4,1,self.n_trunk_out])}}} # Dictionary Containing the No of Neurons for each Layer
        self.act_funcs           = {'DeepONet': {'Branch': {'Main': ['tanh','tanh','linear']},  
                                                  'Shift': {'Main': ['tanh','tanh','tanh','linear']}, 
                                                'Stretch': {'Main': ['tanh','tanh','tanh','linear']}, 
                                               'Rotation': {'Main': ['tanh','tanh','tanh','linear']}, 
                                                  'Trunk': {'Main': ['tanh','exponential','linear']}}}       # Dictionary Containing the Activation Funct.s for each Layer
        self.reg_coeffs          = {'DeepONet': {'Branch': {'Main': [1.e-8,1.e-8]},
                                                  'Shift': {'Main': [1.e-8,1.e-8]},  
                                                'Stretch': {'Main': [1.e-8,1.e-8]},  
                                               'Rotation': {'Main': [1.e-8,1.e-8]},  
                                                  'Trunk': {'Main': [1.e-8,1.e-8]}}}  
        self.dropout_rate        = {'DeepONet': {'Branch': {'Main': None},
                                                  'Shift': {'Main': 5.e-5},  
                                                'Stretch': {'Main': 5.e-5},  
                                               'Rotation': {'Main': 5.e-5},  
                                                  'Trunk': {'Main': None}}}                                # Dictionary Containing the Dropout Rate for each Sub-Component
        self.dropout_pred_flg    = {'DeepONet': {'Branch': {'Main': False},  
                                                  'Shift': {'Main': False},
                                                'Stretch': {'Main': False},
                                               'Rotation': {'Main': False},
                                                  'Trunk': {'Main': False}}}                                 # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        self.dotlayer_bias_flg   = {'DeepONet': False}
        self.softmax_flg         = {'DeepONet': {'Branch': {'Main': False},  
                                                  'Shift': {'Main': False},
                                                'Stretch': {'Main': False},
                                               'Rotation': {'Main': False},
                                                  'Trunk': {'Main': False}}}                                 # Dictionary Containing the Softmax Flag for each Sub-Component 

        #=======================================================================================================================================
        ### Losses
        ## Fully Data Driven
        self.losses              = {'pts': {'name': 'MSE', 'axis': 0}}                                       # Dictionary Containing Loss Functions for Each Data Type
        self.loss_weights        = {'pts': 1.}                                                               # Dictionary Containing Weights for Each Data Type
        # ## Physics Informed
        # self.losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}}    # Dictionary Containing Loss Functions for Each Data Type
        # self.loss_weights        = {'ics': 1., 'res': 1.}                                                    # Dictionary Containing Weights for Each Data Type
        self.metrics             = None                                                                      # List of Metrics                                                          

        #=======================================================================================================================================
        ### Training Quanties
        self.trainable_flg       = {'DeepONet': 'all'}                                                       # Dictionary Containing Instructions for Training Components ('all'/'none'/'only_last')
        self.transfer_flg        = False                                                                     # Flag for Transfer Learning
        self.path_to_transf_fld  = ''                                                                        # Path to Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch             = 20000                                                                     # Number of Epoches
        self.batch_size          = 2048                                                                        # Mini-Batch Size
        self.valid_batch_size    = 2048                                                                        # Validation Mini-Batch Size
        self.lr                  = 1.e-3                                                                     # Initial Learning Rate
        self.lr_decay            = ["exponential", 10000, 0.98]                                              # Instructions for Learning Rate Decay
        self.optimizer           = 'adam'                                                                    # Optimizer
        self.optimizer_params    = [0.9, 0.999, 1e-07]                                                       # Parameters for the Optimizer
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
