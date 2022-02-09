import os
import numpy                                  as np


#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNet_fld):

        self.NRODs               = 7
        self.i_red                = 1
        POD_NAME                 = str(self.i_red) #'All'
        #POD_NAME                 = 'All'

        self.n_modes               = 24



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
        self.path_to_run_fld        = self.ROMNet_fld   + '/../0DReact_Isobaric_500Cases_Up_POD_'+POD_NAME+'_Trunk/'           # Path To Training Folder
        self.path_to_load_fld       = None                                                                   # Path To Pre-Trained Model Folder
        self.path_to_data_fld       = self.ROMNet_fld   + '/../Data/0DReact_Isobaric_500Cases_Up/'+str(self.NRODs)+'PC/OneByOne/POD_'+POD_NAME+'/Trunk/'           # Path To Training Data Folder 
        #self.path_to_data_fld       = self.ROMNet_fld   + '/../Data/0DReact_Isobaric_500Cases_Simple/'+str(self.NRODs)+'PC/All/POD_All/Trunk/'           # Path To Training Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.phys_system          = 'MassSpringDamper'                                                        # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.data_type            = 'PDE'                                                                    # Module to Be Used for Reading Data
        self.generate_flg         = False
        self.n_train              = {'pts': 64}                                                         # No of Training Cases
        # self.n_train              = {'ics': 64, 'res': 128}                                                  # No of Training Cases
        self.valid_perc           = 20.0                                                                     # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.data_dist            = 'uniform'                                                                # Distribution for Sampling Independent Variables
        self.n_test               = 2                                                                        # No of Test Cases
        self.test_flg             = False

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type       = 'FNN'                                                                  # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        self.InputVars           = ['t']                                                                 # List Containing the Input Data Column Names 
        self.OutputVars          = ['POD_'+str(i_mode+1) for i_mode in range(self.n_modes)]                                                             # List Containing the Output Data Column Names
        self.TransFun            = {'log': ['t']} 
        self.NormalizeInput      = True                                                                   # Flag for Normalizing Input Data
        self.Layers              = [np.array([100,100,100,self.n_modes])]                                           # List Containing the No of Neurons per Each NN's Layer
        self.ActFun              = [['sigmoid','sigmoid','sigmoid','linear']]                                 # List Containing the Activation Funct.s per Each NN's Layer
        self.DropOutRate         = 1.e-15                                                                  # NN's Layers Dropout Rate
        self.DropOutPredFlg      = False                                                                  # Flag for Using NN's Dropout during Prediction
        self.NormalizeOutput     = False      

        self.ULayers             = [np.array([100])]
        self.UActFun             = [['tanh']]
        self.VLayers             = [np.array([100])]
        self.VActFun             = [['tanh']]

        #=======================================================================================================================================
        ### Training Quanties
        self.transfer_flg         = False                                                                  # Flag for Using Transfer Learning
        self.path_to_transf_fld      = ''                                                                     # Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch              = 100000                                                                 # Number of Epoches
        self.batch_size           = 64                                                                    # Batch Size for Training
        self.valid_batch_size      = 64                                                                    # Batch Size for Validation
        self.losses              = {'pts': {'name': 'MSE', 'axis': 0}} # Loss Functions
        self.loss_weights         = {'pts': 1.}  
        # self.losses              = {'ics': {'name': 'MSE', 'axis': 0}, 'res': {'name': 'MSE', 'axis': 0}} # Loss Functions
        # self.loss_weights         = {'ics': 1., 'res': 1.}     
        self.metrics             = None                                                                   # List of Metric Functions
        self.lr                  = 5.e-4                                                                 # Initial Learning Rate
        self.lr_decay             = ["exponential", 10000, 0.98]
        self.optimizer           = 'adam'                                                                 # Optimizer
        self.optimizer_params     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-9, 1.e-9], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
        self.callbacks_dict           = {
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
