import os
import numpy      as np
import tensorflow as tf

#=======================================================================================================================================
class inputdata(object):


    def __init__(self, WORKSPACE_PATH, ROMNet_fld):

        self.NRODs               = 3

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
        self.path_to_run_fld        = self.ROMNet_fld   + '/../PSR_100Cases/'                                 # Path To Training Folder
        self.path_to_load_fld       = self.ROMNet_fld   + '/../PSR_100Cases/DeepONet/Deterministic/Run_1/'                            # Path To Pre-Trained Model Folder
        self.ROMPred_Flg         = True
        self.path_to_data_fld       = self.ROMNet_fld   + '/../Data/PSR_100Cases/'+str(self.NRODs)+'PC/'            # Path To Training Data Folder 
        # self.path_to_data_fld       = self.ROMNet_fld   + '/../Data/PSR_100Cases/Orig/'                        # Path To Training Data Folder 

        #=======================================================================================================================================
        ### Physical System
        self.phys_system          = 'PSR'                                                                  # Name of the Physical System for PINN

        #=======================================================================================================================================
        ### Data
        self.data_type            = 'PDE'                                                                    # Module to Be Used for Reading Data
        self.generate_flg         = False
        self.n_train              = {'ics': 64, 'res': 128, 'pts': 128}                                                # No of Training Cases
        # self.n_train              = {'res': 128}                                                             # No of Training Cases
        # self.n_train              = {'ics': 64, 'pts': 128}                                                             # No of Training Cases
        # self.n_train              = {'pts': 64}                                                             # No of Training Cases
        self.valid_perc           = 20.0                                                                     # Percentage of Training Data to Be Used for Validation (e.g., = 20.0 => 20%)
        self.data_dist            = 'uniform'                                                                # Distribution for Sampling Independent Variables
        self.n_test               = 2                                                                        # No of Test Cases
        self.test_flg             = False

        #=======================================================================================================================================
        ## NN Model Structure
        self.surrogate_type       = 'DeepONet'                                                             # Type of Surrogate ('DeepONet' / 'FNN' / 'FNN-SourceTerms')
        self.ProbApproach        = 'Deterministic'                                                        # Probabilistic Technique for Training the BNN (if Any)
        self.NormalizeInput      = False                                                                   # Flag for Normalizing Branch's Input Data
        # self.BranchToTrunk       = range(self.NRODs)                                                                # Index of the Trunk Corresponding to i-th Branch
        self.BranchToTrunk       = [0]*self.NRODs                                                                # Index of the Trunk Corresponding to i-th Branch
        self.BranchVars          = ['log10(Rest)']                                                        # List Containing the Branch's Input Data Column Names
        self.BranchLayers        = [np.array([16,32,64])]*self.NRODs                                            # List Containing the No of Neurons per Each Branch's Layer
        self.BranchActFun        = [['sigmoid','sigmoid','sigmoid']]*self.NRODs                                             # List Containing the Activation Funct.s per Each Branch's Layer
        self.BranchDropOutRate   = 1.e-10                                                                 # Branch's Layers Dropout Rate
        self.BranchDropOutPredFlg= False                                                                  # Flag for Using Branch's Dropout during Prediction
        self.BranchSoftmaxFlg    = False                                                                  # Flag for Using Softmax after Branch's Last Layer
        self.TrunkVars           = ['t']                                                                  # List Containing the Trunk's Input Data Column Names
        self.TrunkLayers         = [np.array([64,64,64])]#*self.NRODs                                                # List Containing the No of Neurons per Each Trunk's Layer
        self.TrunkActFun         = [['tanh','tanh','tanh']]#*self.NRODs                                             # List Containing the Activation Funct.s per Each Trunk's Layer
        self.TrunkDropOutRate    = 1.e-10                                                                  # Trunk's Layers Dropout Rate  
        self.TrunkDropOutPredFlg = False                                                                  # Flag for Using Trunk's Dropout during Prediction
        self.TransFun            = {'log': ['t']} 
        self.FinalLayerFlg       = True                                                                   # Flag for Using a Full Linear Layer after Dot-Product Layer
        self.OutputVars          = ['PC_'+str(i+1) for i in range(self.NRODs)]
        self.NormalizeOutput     = True                                                                   # Flag for Normalizing Branch's Input Data

        #=======================================================================================================================================
        ### Training Quanties
        self.transfer_flg         = False                                                                  # Flag for Using Transfer Learning
        self.path_to_transf_fld      = ''                                                                     # Folder Containing the Trained Model to be Used for Transfer Learning 
        self.n_epoch              = 100000                                                                   # Number of Epoches
        self.batch_size           = 64                                                                     # Mini-Batch Size
        self.valid_batch_size      = 64                                                                   # Validation Mini-Batch Size
        self.RunEagerlyFlg       = True
        self.losses              = {'ics': {'name': 'mse', 'axis': 0}, 'res': {'name': 'mse', 'axis': 0}, 'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        self.loss_weights         = {'ics': 1.e0, 'res': 1.e-7, 'pts': 1.e0}     
        # self.losses              = {'res': {'name': 'mse', 'axis': 0}}                                    # Loss Functions
        # self.loss_weights         = {'res': 1.} 
        # self.losses              = {'ics': {'name': 'mse', 'axis': 0}, 'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        # self.loss_weights         = {'ics': 1., 'pts': 10.}   
        # self.losses              = {'pts': {'name': 'mse', 'axis': 0}} # Loss Functions
        # self.loss_weights         = {'pts': 1.}     
        self.metrics             = None                   
        self.lr                  = 1.e-5                                                          # Initial Learning Rate
        self.lr_decay             = ["exponential", 5000, 0.98]
        self.optimizer           = 'adam'                                                                 # Optimizer
        self.optimizer_params     = [0.9, 0.999, 1e-07]                                                    # Parameters for the Optimizer
        self.WeightDecay         = np.array([1.e-6,1.e-6], dtype=np.float64)                             # Hyperparameters for L1 and L2 Weight Decay Regularizations
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
