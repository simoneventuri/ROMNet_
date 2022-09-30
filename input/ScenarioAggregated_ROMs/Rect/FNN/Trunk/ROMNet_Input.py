import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH):

        with open('./iVar.csv') as f:
            for line in f: # read rest of lines
                iVar = int(line)

        self.DRType               = 'OneByOne'
        self.DRAlog               = 'PCA'
        self.NRODs                = 128
        self.iVar                 = iVar


        #=======================================================================================================================================
        ### Case Name
        self.run_idx              = 0                                                                         # Training Case Identification Number. If ==0, the code automatically assigns one.

        #=======================================================================================================================================
        ### Execution Flags
        self.train_int_flg        = 1                                                                         # Integer Flag for Training (0=>No, 1=>Yes)

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH       = WORKSPACE_PATH                                                            # os.getenv('WORKSPACE_PATH')      
        self.ROMNet_fld           = self.WORKSPACE_PATH + '/ROMNet/romnet/'                                   # $WORKSPACE_PATH/ROMNet/romnet/
        self.path_to_run_fld      = self.ROMNet_fld   + '/../Rect_200Instants_TransRotScale_'+str(self.NRODs)+self.DRAlog+'/Var'+str(self.iVar)+'_Trunk/'           # Path To Training Folder
        self.path_to_load_fld     = None                                                                      # Path To Pre-Trained Model Folder

        #=======================================================================================================================================
        ### Physical System
        self.phys_system          = 'Generic'                                                                 # Name of the Physical System for PINN

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
        # self.n_test               = 2    

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type      = 'FNN'                                                                # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.plot_graph_flg      = True                                                                      # Flag for Plotting and Saving the Graph for the Network Structure
        self.trans_fun           = None #{'log': ['t']}                                                            # Dictionary Containing Functions to Be Applied to Input Data 
        self.data_preproc_type   = 'range'
        self.norm_input_flg      = {'FNN': {'FNN': True}}                                           # Dictionary Containing Flags for Normalizing Input Data for each Component
        self.norm_output_flg     = True                                                                      # Flag for Normalizing Output Data
        self.rectify_flg         = False
        self.internal_pca_flg    = False

        # -----------------------------------------------------------------------------------
        self.ROM_pred_flg        = False
        self.path_to_data_fld    = self.ROMNet_fld   + '/../Data/Rect_200Instants_TransRotScale/'+str(self.NRODs)+self.DRAlog+'/'+self.DRType+'/'+'/Var'+str(self.iVar)+'/Trunk/'      # Path To Training Data Folder 
        self.Vars                = ['t_'+str(i+1) for i in range(self.NRODs)]
        self.output_vars         = self.Vars                                                                             # List Containing the Output Data Variable Names for each System
        self.input_vars_all      = ['x','y']                                                                  # List Containing all the Input Data Variable Names
        self.input_vars          = {'FNN': {'FNN': ['x','y']}}                                                       # Dictionary Containing the Input  Data Variable Names for each Component
        
        # -----------------------------------------------------------------------------------
        self.structure               = {'FNN': {}}
        self.structure['FNN']['FNN'] = ['Main']
        self.n_neurons               = {'FNN': {'FNN': {'Main': np.array([128,128,128,128,128,self.NRODs])}}} # Dictionary Containing the No of Neurons for each Layer
        self.act_funcs               = {'FNN': {'FNN': {'Main': ['tanh','tanh','tanh','tanh','tanh','linear']}}}       # Dictionary Containing the Activation Funct.s for each Layer
        self.dropout_rate            = {'FNN': {'FNN': {'Main': None}}}                                  # Dictionary Containing the Dropout Rate for each Sub-Component
        self.dropout_pred_flg        = {'FNN': {'FNN': {'Main': False}}}                                 # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        self.softmax_flg             = {'FNN': {'FNN': {'Main': False}}}                                   # Dictionary Containing the Softmax Flag for each Sub-Component 


        #=======================================================================================================================================
        ### Losses
        ## Fully Data Driven
        self.losses              = {'pts': {'name': 'MSE', 'axis': 0}}                                       # Dictionary Containing Loss Functions for Each Data Type
        self.loss_weights        = {'pts': 1.}                                                               # Dictionary Containing Weights for Each Data Type
        self.run_eagerly_flg     = False
        # ## Physics Informed
        # self.losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}}    # Dictionary Containing Loss Functions for Each Data Type
        # self.loss_weights        = {'ics': 1.e-1, 'res': 1.}                                                 # Dictionary Containing Weights for Each Data Type
        # self.run_eagerly_flg     = True
        self.metrics             = None                                                                      # List of Metrics                                                          

        #=======================================================================================================================================
        ### Training Quanties
        self.trainable_flg       = {'DeepONet': 'all'}                                                       # Dictionary Containing Instructions for Training Components ('all'/'none'/'only_last')
        self.transfer_flg        = False                                                                     # Flag for Transfer Learning
        self.path_to_transf_fld  = ''                                                                        # Path to Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch             = 100000                                                                    # Number of Epoches
        self.batch_size          = 128                                                                       # Mini-Batch Size
        self.valid_batch_size    = 128                                                                       # Validation Mini-Batch Size
        self.lr                  = 1.e-3                                                                     # Initial Learning Rate
        self.lr_decay            = ["exponential", 10000, 0.98]                                              # Instructions for Learning Rate Decay
        self.optimizer           = 'adam'                                                                    # Optimizer
        self.optimizer_params    = [0.9, 0.999, 1e-07]                                                       # Parameters for the Optimizer
        self.weight_decay_coeffs = np.array([1.e-9, 1.e-9], dtype=np.float64)                              # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.callbacks_dict           = {
            'base': {
                'stateful_metrics': None
            },
            'early_stopping': {
                'monitor':              'val_tot_loss',
                'min_delta':            1.e-8,
                'patience':             6000,
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
            #     'loss_weights0': {'res': 1.e0},
            #     'freq':          2,
            #     'shape_1':       1
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
