import os
import numpy                                  as np
import pandas                                 as pd 

#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH):

        self.NRODs               = 20

        #=======================================================================================================================================
        ### Case Name
        self.run_idx             = 0                                                                      # Training Case Identification Number 

        #=======================================================================================================================================
        ### Execution Flags
        self.train_int_flg       = 1                                                                         # Integer Flag for Training (0=>No, 1=>Yes)

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                                                                # os.getenv('WORKSPACE_PATH')       
        self.ROMNet_fld          = self.WORKSPACE_PATH + '/ROMNet/romnet/'                                       # $WORKSPACE_PATH/ROMNet/romnet/
        self.path_to_data_fld    = self.ROMNet_fld   + '/../Data/0DReact_Isobaric_500Cases_CH4_/20PC/ROM/Decoder/'                # Path To Training Data Folder 
        self.path_to_run_fld     = self.ROMNet_fld + '/../0DReact_Isobaric_500Cases_CH4_Decoder_/'                       # Path To Training Folder
        self.path_to_load_fld    = None #self.ROMNet_fld + '/../Data/0DReact_Isobaric_500Cases/Orig/OneByOne/FNN/Final.h5'    # Path To Pre-Trained Model Folder 
        #self.path_to_load_fld    = self.ROMNet_fld +'/../0DReact_Isobaric_500Cases/DeepONet/8Modes/'            # Path To Pre-Trained Model Folder 

        #=======================================================================================================================================
        ### Physical System
        self.phys_system         = 'Generic'                                                        # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.data_type           = 'PDE'                                                                     # Module to Be Used for Reading Data
        self.generate_flg        = False                                                                     # Flag for Generating Data
        ## Fully Data Driven
        self.n_train             = {'pts': 0}                                                                # Type/No of Data Points
        # ## Physics Informed
        # self.n_train             = {'ics': 0, 'res': 0}                                                      # Type/No of Data Points
        # self.valid_perc          = 20.0                                                                    # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        # self.data_dist           = 'uniform'                                                               # Distribution for Sampling Independent Variables
        # self.test_flg             = False                                                                  # Test Flag
        # self.n_test               = 2                                                                      # No of Test Cases

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type      = 'FNN'                                                                     # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.plot_graph_flg      = True                                                                      # Flag for Plotting and Saving the Graph for the Network Structure

        FileName                 = self.path_to_data_fld+'/../../Vars.csv'
        self.input_vars_all      = list(pd.read_csv(FileName, delimiter=',', header=None).to_numpy()[0,:])
        #self.input_vars_all      = ['PC_'+str(iPC+1) for iPC in range(self.NRODs)]
        self.input_vars          = {'FNN': {'FNN': self.input_vars_all}}                                     # Dictionary Containing the Input  Data Variable Names for each Component
        FileName                 = self.path_to_data_fld+'/../../CleanVars_ToRed.csv'
        self.output_vars         = list(pd.read_csv(FileName, delimiter=',', header=None).to_numpy()[0,:])
        self.n_outputs           = len(self.output_vars)
        self.trans_fun           = None #{'log': ['t']}                                                      # Dictionary Containing Functions to Be Applied to Input Data 
        self.norm_input_flg      = {'FNN': {'FNN': True}}                                                   # Dictionary Containing Flags for Normalizing Input Data for each Component
        self.norm_output_flg     = True                                                                    # Flag for Normalizing Output Data
        self.data_preproc_type   = 'range'
        self.structure           = {'FNN': {'FNN': ['Main']}}                                                # Dictionary Containing the Structure of the Network
        self.n_neurons           = {'FNN': {'FNN': {'Main': np.array([25,30,40,self.n_outputs,self.n_outputs])}}}           # Dictionary Containing the No of Neurons for each Layer
        self.act_funcs           = {'FNN': {'FNN': {'Main': ['relu','relu','relu','relu','softplus']}}}               # Dictionary Containing the Activation Funct.s for each Layer
        self.dropout_rate        = {'FNN': {'FNN': {'Main': 1.e-8}}}                                         # Dictionary Containing the Dropout Rate for each Sub-Component
        self.dropout_pred_flg    = {'FNN': {'FNN': {'Main': False}}}                                         # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        self.softmax_flg         = {'FNN': False}                                         # Dictionary Containing the Softmax Flag for each Sub-Component 
        self.rectify_flg         = False

        #=======================================================================================================================================
        ### Losses
        ## Fully Data Driven
        self.losses              = {'pts': {'name': 'MSE', 'axis': 0}}                                       # Dictionary Containing Loss Functions for Each Data Type
        self.loss_weights        = {'pts': 1.}                                                               # Dictionary Containing Weights for Each Data Type
        ### Physics Informed
        # self.losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}}    # Dictionary Containing Loss Functions for Each Data Type
        # self.loss_weights        = {'ics': 1., 'res': 1.}                                                    # Dictionary Containing Weights for Each Data Type
        self.metrics             = None     

        #=======================================================================================================================================
        ### Training Quanties
        self.trainable_flg       = {'FNN': 'all'}                                                            # Dictionary Containing Instructions for Training Components ('all'/'none'/'only_last')
        self.transfer_flg        = False                                                                     # Flag for Transfer Learning
        self.path_to_transf_fld  = ''                                                                        # Path to Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch             = 100000                                                                    # Number of Epoches
        self.batch_size          = 4096                                                                        # Batch Size for Training
        self.valid_batch_size    = 4096                                                                       # Batch Size for Validation
        self.lr                  = 1.e-3                                                                     # Initial Learning Rate
        self.lr_decay            = ["exponential", 10000, 0.98]                                               # Instructions for Learning Rate Decay
        self.optimizer           = 'adam'                                                                    # Optimizer
        self.optimizer_params    = [0.9, 0.999, 1e-07]                                                       # Parameters for the Optimizer
        self.weight_decay_coeffs = np.array([1.e-10, 1.e-10], dtype=np.float64)                                # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.callbacks_dict      = {
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
