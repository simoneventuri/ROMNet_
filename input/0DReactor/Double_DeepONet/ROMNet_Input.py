import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNet_fld):

        self.NRODs               = 7
        self.i_redSel             = range(self.NRODs)
        self.NRODsSel            = len(self.i_redSel)

        self.n_modes               = 1

        #=======================================================================================================================================
        ### Case Name
        self.run_idx            = 0                                                                      # Training Case Identification Number 

        #=======================================================================================================================================
        ### Execution Flags
        self.train_int_flg       = 2                                                                      # Training                       0=>No, 1=>Yes
        self.plot_int_flg        = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training
        self.PredictIntFlg       = 2                                                                      # Plotting Data                  0=>Never, 1=>After Training, 2=>Also During Training

        #=======================================================================================================================================
        ### Paths
        self.WORKSPACE_PATH      = WORKSPACE_PATH                                                         # os.getenv('WORKSPACE_PATH')      
        self.ROMNet_fld          = ROMNet_fld                                                             # $WORKSPACE_PATH/ProPDE/
        self.path_to_run_fld        = self.ROMNet_fld   + '/../0DReact_Isobaric_1Cond/'                                 # Path To Training Folder
        self.ROMPred_Flg         = True
        self.path_to_data_fld       = self.ROMNet_fld   + '/../Data/0DReact_Isobaric_1Cond/'+str(self.NRODs)+'PC/'            # Path To Training Data Folder 
        self.path_to_load_fld       = '/Users/sventur/WORKSPACE/ROMNet/0DReact_Isobaric_1Cond/DeepONet/Deterministic/Run_8/'            # Path To Training Data Folder 
        #self.path_to_load_fld       = self.ROMNet_fld   + '/../Data/0DReact_Isobaric_500Cases_Simple/10PC/All/FNN/Final.h5'            # Path To Training Data Folder 
        
        #=======================================================================================================================================
        ### Data
        self.phys_system          = 'ZeroDR'                                                             # Module to Be Used for Reading Data

        #=======================================================================================================================================
        ### Data
        self.data_type            = 'PDE'                                                                    # Module to Be Used for Reading Data
        self.generate_flg         = False
        # self.n_train              = {'scs': 64, 'res': 128, 'pts': 128}                                                # No of Training Cases
        # self.n_train              = {'res': 128}                      x                                       # No of Training Cases
        # self.n_train              = {'scs': 64, 'pts': 128}                                                             # No of Training Cases
        self.n_train              = {'pts': 64}                                                             # No of Training Cases
        self.valid_perc           = 20.0                                                                     # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.data_dist            = 'uniform'                                                                # Distribution for Sampling Independent Variables
        self.n_test               = 2                                                                        # No of Test Cases
        self.test_flg             = False

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type       = 'Double_DeepONet'                                                       # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.n_deeponets         = 1
        self.ProbApproach        = 'Deterministic'                                                         # Probabilistic Technique for Training the BNN (if Any)
        self.trans_fun           = {'log': ['t']}                                                          # Dictionary Containing Functions to Be Applied to Input Data 
        self.t_scale             = 'log'
        self.norm_output_flg     = True                                                                    # Flag for Normalizing Output Data
        self.output_vars         = ['PC_'+str(i+1) for i in self.i_redSel]                                  # List Containing the Output Data Variable Names for each System
        self.input_vars_all      = ['PC0_'+str(i+1) for i in range(self.NRODs)]+['t']                      # List Containing all the Input Data Variable Names
        self.input_vars          = {  'DeepONet': {'Branch': ['PC0_'+str(i+1) for i in range(self.NRODs)],
                                                    'Rigid': ['PC0_'+str(i+1) for i in range(self.NRODs)],
                                                    'Trunk': ['t']},
                                    'DeepONet_2': {'Branch': ['PC0_'+str(i+1) for i in range(self.NRODs)],
                                                    'Rigid': ['PC0_'+str(i+1) for i in range(self.NRODs)],
                                                    'Trunk': ['t']}}                                         # Dictionary Containing the Input  Data Variable Names for each Component
        self.branch_to_trunk     = {'DeepONet': 'unstacked', 'DeepONet_2': 'unstacked'}
        self.norm_input_flg      = {  'DeepONet': {'Branch': False, 
                                                    'Rigid': False,
                                                    'Trunk': False},
                                    'DeepONet_2': {'Branch': False, 
                                                    'Rigid': False,
                                                    'Trunk': False}}                                          # Dictionary Containing Flags for Normalizing Input Data for each Component
        self.structure                                      = {'DeepONet': {},'DeepONet_2': {}}
        for i in range(self.NRODsSel):  
           self.structure['DeepONet']['Branch_'+str(i+1)]   = ['Main']
        self.structure['DeepONet']['Rigid']                 = ['Main']
        for i in range(self.NRODsSel):  
           self.structure['DeepONet']['Trunk_'+str(i+1)]    = ['Main']    
        for i in range(self.NRODsSel):
           self.structure['DeepONet_2']['Branch_'+str(i+1)] = ['Main']
        self.structure['DeepONet_2']['Rigid']               = ['Main']
        for i in range(self.NRODsSel):
           self.structure['DeepONet_2']['Trunk_'+str(i+1)]  = ['Main']                                       # Dictionary Containing the Structure of the Network
        self.n_neurons           = {  'DeepONet': {'Branch': {'Main': np.array([32,32,32,self.n_modes+2])},
                                                    'Rigid': {'Main': np.array([32,32,32,self.NRODsSel])},
                                                    'Trunk': {'Main': np.array([32,32,32,self.n_modes])}},
                                    'DeepONet_2': {'Branch': {'Main': np.array([32,32,32,self.n_modes+2])},
                                                    'Rigid': {'Main': np.array([32,32,32,self.NRODsSel])},
                                                    'Trunk': {'Main': np.array([32,32,32,self.n_modes])}}}     # Dictionary Containing the No of Neurons for each Layer
        self.act_funcs           = {  'DeepONet': {'Branch': {'Main': ['tanh','tanh','tanh','linear']},
                                                    'Rigid': {'Main': ['tanh','tanh','tanh','linear']},
                                                    'Trunk': {'Main': ['tanh','tanh','tanh','linear']}},
                                    'DeepONet_2': {'Branch': {'Main': ['tanh','tanh','tanh','linear']},
                                                    'Rigid': {'Main': ['tanh','tanh','tanh','linear']},
                                                    'Trunk': {'Main': ['tanh','tanh','tanh','linear']}}}     # Dictionary Containing the Activation Funct.s for each Layer
        self.dropout_rate        = {  'DeepONet': {'Branch': {'Main': None},
                                                    'Rigid': {'Main' :None},
                                                    'Trunk': {'Main': None}},
                                    'DeepONet_2': {'Branch': {'Main': None},
                                                    'Rigid': {'Main' :None},
                                                    'Trunk': {'Main': None}}}                              # Dictionary Containing the Dropout Rate for each Sub-Component
        self.dropout_pred_flg    = {  'DeepONet': {'Branch': {'Main': False},
                                                    'Trunk': {'Main': False}},
                                    'DeepONet_2': {'Branch': {'Main': False},
                                                    'Trunk': {'Main': False}}}                               # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        self.softmax_flg         = {  'DeepONet': {'Branch': {'Main': False},
                                                    'Trunk': {'Main': False}},
                                    'DeepONet_2': {'Branch': {'Main': False},
                                                    'Trunk': {'Main': False}}}                               # Dictionary Containing the Softmax Flag for each Sub-Component 
        # self.structure                                  = {'DeepONet': {}}
        # for i in range(self.NRODs):
        #    self.structure['DeepONet']['Branch_'+str(i+1)] = ['Main','U','V']
        # self.structure['DeepONet']['Rigid']               = ['Main']
        # for i in range(self.NRODs):
        #    self.structure['DeepONet']['Trunk_'+str(i+1)]  = ['Main','U','V']                               # Dictionary Containing the Structure of the Network
        # self.n_neurons           = {'DeepONet': {'Branch': {'Main': np.array([32,32,32,self.n_modes+2]),
        #                                                        'U': np.array([32]),
        #                                                        'V': np.array([32])},
        #                                           'Rigid': {'Main': np.array([32,32,32,self.n_modes])},
        #                                           'Trunk': {'Main': np.array([32,32,32,self.n_modes+2]),
        #                                                        'U': np.array([32]),
        #                                                        'V': np.array([32])}}}                      # Dictionary Containing the No of Neurons for each Layer
        # self.act_funcs           = {'DeepONet': {'Branch': {'Main': ['sigmoid','sigmoid','sigmoid','linear'],
        #                                                        'U': ['tanh'],
        #                                                        'V': ['tanh']},
        #                                           'Rigid': {'Main': ['sigmoid','sigmoid','sigmoid','linear']},
        #                                           'Trunk': {'Main': ['sigmoid','sigmoid','sigmoid','linear'],
        #                                                        'U': ['tanh'],
        #                                                        'V': ['tanh']}}}                            # Dictionary Containing the Activation Funct.s for each Layer
        # self.dropout_rate        = {'DeepONet': {'Branch': {'Main':  1.e-3, 'U': None,  'V': None},
        #                                           'Rigid': {'Main': None},
        #                                           'Trunk': {'Main':  1.e-3}}}                              # Dictionary Containing the Dropout Rate for each Sub-Component
        # self.dropout_pred_flg    = {'DeepONet': {'Branch': {'Main': False,  'U': False, 'V': False}}}      # Dictionary Containing the Dropout-at-Prediction Flag for each Sub-Component 
        # self.softmax_flg         = {'DeepONet': {'Branch': {'Main': False}}}                               # Dictionary Containing the Softmax Flag for each Sub-Component 


        #=======================================================================================================================================
        ### Training Quanties
        self.trainable_flg       = {'DeepONet': 'all', 'DeepONet_2': 'all'}
        self.transfer_flg         = False                                                                  # Flag for Using Transfer Learning
        self.path_to_transf_fld      = None
        self.n_epoch              = 100000                                                                   # Number of Epoches
        self.batch_size           = 64                                                                     # Mini-Batch Size
        self.valid_batch_size      = 64                                                                 # Validation Mini-Batch Size
        self.RunEagerlyFlg       = False
        # self.losses              = {'scs': {'name': 'mse', 'axis': 0}, 'res': {'name': 'mse', 'axis': 0}, 'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        # self.loss_weights         = {'scs': 1.e-1, 'res': 1.e-8, 'pts': 1.e0}     
        # self.losses              = {'res': {'name': 'mse', 'axis': 0}}                                    # Loss Functions
        # self.loss_weights         = {'res': 1.} 
        # self.losses              = {'scs': {'name': 'mse', 'axis': 0}, 'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        # self.loss_weights         = {'scs': 0.1, 'pts': 1.}   
        self.losses              = {'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        self.loss_weights         = {'pts': 1.}     
        self.metrics             = None                   
        self.lr                  = 1.e-4                                                          # Initial Learning Rate
        self.lr_decay             = ["exponential", 100000, 0.98]
        self.optimizer           = 'adam'                                                                 # Optimizer
        self.optimizer_params     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.weight_decay_coeffs = np.array([1.e-8,1.e-8], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
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
