import time
import os
import shutil
import sys
from pathlib                              import Path
import numpy                                  as np
import pandas                                 as pd
from sklearn.model_selection              import train_test_split
import tensorflow                             as tf
from tensorflow                           import train as tf_train

from .model         import Model
from ..training     import LossHistory, get_loss, get_optimizer, callbacks
from ..architecture import load_model_, load_weights_
from ..             import utils
from ..metrics      import get_metric



#===============================================================================
from datetime import datetime

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return get_curr_time()

#===============================================================================



#===============================================================================
class Model_TF(Model):     
    """

    Args:
        
        
    """



    #===========================================================================
    # Class Initialization
    def __init__(self, InputData):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_tf.py    ]:   Initializing the ML Model')

        self.surrogate_type        = InputData.surrogate_type
  
        self.train_int_flg         = InputData.train_int_flg
        self.path_to_load_fld      = InputData.path_to_load_fld
  
        self.got_stats             = False
        try:
            self.norm_output_flg   = InputData.norm_output_flg
        except:
            self.norm_output_flg   = False
        try:
            self.data_preproc_type = InputData.data_preproc_type
        except:
            self.data_preproc_type = None

        if (self.train_int_flg > 0):

            #-------------------------------------------------------------------
            ### Dealing With Paths

            InputData.path_to_run_fld += '/' + self.surrogate_type + '/'
            path = Path(InputData.path_to_run_fld)
            path.mkdir(parents=True, exist_ok=True)

            Prefix = 'Run_'
            if (InputData.run_idx == 0):
                if (len([x for x in os.listdir(InputData.path_to_run_fld) 
                                                          if 'Run_' in x]) > 0):
                    InputData.run_idx = str(np.amax( np.array( [int(x[len(Prefix):]) for x in os.listdir(InputData.path_to_run_fld) if Prefix in x], dtype=int) ) + 1)
                else:
                    InputData.run_idx = 1

            InputData.TBCheckpointFldr = InputData.path_to_run_fld + '/TB/' +     \
               Prefix + str(InputData.run_idx) + "_{}".format(get_start_time())

            InputData.path_to_run_fld    += '/' + Prefix + str(InputData.run_idx)

            self.path_to_run_fld      = InputData.path_to_run_fld
            self.TBCheckpointFldr  = InputData.TBCheckpointFldr

            # Creating Folders
            path = Path( self.path_to_run_fld+'/Figures/' )
            path.mkdir(parents=True, exist_ok=True)

            path = Path( self.path_to_run_fld+'/Model/Params' )
            path.mkdir(parents=True, exist_ok=True)

            path = Path( self.path_to_run_fld+'/Training/Params' )
            path.mkdir(parents=True, exist_ok=True)
                     
            print("\n[ROMNet - model_tf.py    ]:   Trained Model can be Found here: "    + 
                                                              self.path_to_run_fld)
            print("\n[ROMNet - model_tf.py    ]:   TensorBoard Data can be Found here: " + 
                                                          self.TBCheckpointFldr)

            # Copying Input File
            print("\n[ROMNet - model_tf.py    ]:   Copying Input File From: " + \
                              InputData.InputFilePath+'/ROMNet_Input.py to ' + \
                                         self.path_to_run_fld + '/ROMNet_Input.py')
            shutil.copyfile(InputData.InputFilePath+'/ROMNet_Input.py', 
                                         self.path_to_run_fld + '/ROMNet_Input.py')

            #-------------------------------------------------------------------

        else:

            self.path_to_run_fld = InputData.path_to_run_fld

    #===========================================================================



    #===========================================================================
    @utils.timing
    def build(
        self,
        InputData,
        data,
        Net,
        system,
        loadfile_no=None,
    ):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_tf.py    ]:   Building the ML Model')

        self.data        = data


        #-----------------------------------------------------------------------
        if (InputData.transfer_flg): 
            self.NN_Transfer_Model = load_weights_(InputData.path_to_transf_fld)
        else:
            self.NN_Transfer_Model = None

        if (self.train_int_flg > 0):
            self.norm_input  = self.data.norm_input
            self.stat_input  = self.data.stat_input
            self.stat_output = self.data.stat_output
        else:
            try:
                self.read_data_statistics()
            except:
                self.stat_input  = None
                self.stat_output = None
            self.norm_input  = None
            
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.net = Net(InputData, self.norm_input, None, self.stat_output, system)

        self.net.AntiPCA_flg = False
        # try:
        #     self.net.AntiPCA_flg = not data.system.ROM_pred_flag
        # else:
        #     self.net.AntiPCA_flg = False

        # if (self.net.AntiPCA_flg):
        #     self.net.A_AntiPCA   = data.system.A
        #     self.net.C_AntiPCA   = data.system.C
        #     self.net.D_AntiPCA   = data.system.D

        self.net.build((1,self.net.n_inputs))
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if (self.train_int_flg == 0):
            if (loadfile_no):
                self.load_params(self.path_to_run_fld + "/Training/Params/"+loadfile_no+".h5")
            else:
                self.load_params(self.path_to_run_fld + "/Training/Params/")
        else:
            if (self.path_to_load_fld is not None):
                if (loadfile_no):
                    self.load_params(self.path_to_load_fld + "/Training/Params/"+loadfile_no+".h5")
                if (self.path_to_load_fld[-1] == '/'):
                    self.load_params(self.path_to_load_fld + "/Training/Params/")
                else:
                    self.load_params(self.path_to_load_fld)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.save_params(self.path_to_run_fld+'/Model/Params/Initial.h5')
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        # TODO: Avoid creating multiple graphs by using tf.TensorSpec.
        @tf.function
        def outputs(self, inputs):
            return self.net.call(inputs)


        def _outputs(self, inputs):
            outs = self.outputs(inputs)
            return utils.to_numpy(outs)

        #-----------------------------------------------------------------------


    #===========================================================================



    #===========================================================================
    @utils.timing
    def compile(self, InputData):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_tf.py    ]:   Compiling the ML Model ... ')


        #-----------------------------------------------------------------------
        print('\n[ROMNet - model_tf.py    ]:   Creating the Models Graph')

        self.graph = self.net.get_graph()

        self.graph.summary()

        with open(self.path_to_run_fld+'/Model/Model_Summary.txt', 'w') as f:
            self.graph.summary(print_fn=lambda x: f.write(x + '\n'))

        try:
            plot_graph_flg = InputData.plot_graph_flg
        except:
            plot_graph_flg = True
        if (plot_graph_flg):            
            tf.keras.utils.plot_model(self.graph, self.path_to_run_fld + 
                                                                 "/Model/Model.png")

            ModelFile = self.path_to_run_fld + '/Model/' + self.net.structure_name + '/'
            self.graph.save(ModelFile)

        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        opt = get_optimizer(InputData)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if (InputData.loss_weights is not None):
            self.loss_weights = {}
            for data_id in InputData.loss_weights.keys():
                self.loss_weights[data_id] =                                   \
                                 list(InputData.loss_weights[data_id].values()) \
                           if isinstance(InputData.loss_weights[data_id], dict) \
                                             else InputData.loss_weights[data_id]
        else:
            self.loss_weights = None
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if (self.norm_output_flg):
            self.save_data_statistics(self.data.stat_input, self.data.stat_output)
            self.data.system.norm_output_flg = True
            self.data.system.output_mean    = self.output_mean[np.newaxis,...]
            self.data.system.output_std     = self.output_std[np.newaxis,...]
            self.data.system.output_min     = self.output_min[np.newaxis,...]
            self.data.system.output_max     = self.output_max[np.newaxis,...]
            self.data.system.output_range   = self.output_range[np.newaxis,...]
        else:
            self.data.system.norm_output_flg = False
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.net.residual  = self.data.res_fn(self.net)    
        self.net.fROM_anti = self.data.fROM_anti
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.loss               = {}
        self.net.data_ids       = []
        self.net.data_ids_valid = []
        for data_id, LossID in InputData.losses.items():
            self.net.data_ids.append(data_id)
            self.net.data_ids_valid.append(data_id)

            if isinstance(LossID, dict) and 'name' in LossID.keys():
                self.loss[data_id] = get_loss(LossID)
            elif isinstance(LossID, dict):
                self.loss[data_id] = get_loss(list(LossID.values()))
            elif isinstance(LossID, (set,list,tuple)):
                self.loss          = get_loss(LossID)

        self.net.data_ids       = tuple(self.net.data_ids)
        self.net.data_ids_valid = tuple(self.net.data_ids_valid)

        if (self.data.Type != 'PDE'):
            self.loss = self.loss['ext']

        self.n_train_tot = self.data.n_train_tot
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if InputData.metrics is not None:
            self.metrics = [ get_metric(met) for met in InputData.metrics ]
        else:
            self.metrics = None
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        try:
            run_eagerly_flg = InputData.run_eagerly_flg
        except:
            run_eagerly_flg = False

        if (run_eagerly_flg):
            print('\n\n\n[ROMNet - model_tf.py    ]:   WARNING: Running Eagerly!\n\n\n')

        self.net.compile(self.data,
                         optimizer    = opt,
                         loss         = self.loss,
                         loss_weights = self.loss_weights,
                         metrics      = self.metrics,
                         run_eagerly  = run_eagerly_flg)
        #-----------------------------------------------------------------------


    #===========================================================================



    #===========================================================================
    @utils.timing
    def train(self, InputData):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_tf.py    ]:   Training the ML Model ... ')

        self.loss_history = LossHistory()
    
        self.n_outputs = self.data.n_outputs
        CallBacksList  = callbacks.get_callback(self, InputData)

        if (InputData.data_type == 'PDE'):
            x        = self.data.train
            y        = None
            xy_valid = self.data.valid 
        else:
            x        = self.data.train['ext'][0]
            y        = self.data.train['ext'][1]
            xy_valid = (self.data.valid['ext'][0], self.data.valid['ext'][1])


        History       = self.net.fit(x=x, 
                                     y=y, 
                                     batch_size=InputData.batch_size, 
                                     validation_batch_size=InputData.valid_batch_size, 
                                     validation_data=xy_valid, 
                                     verbose=1, 
                                     epochs=InputData.n_epoch, 
                                     callbacks=CallBacksList)

        pd.DataFrame.from_dict(History.history).to_csv(
             path_or_buf=self.path_to_run_fld+'/Training/History.csv', 
             index=False
             )
        
        self.loss_history.history = History.history


        self.save_params(self.path_to_run_fld+'/Model/Params/Final.h5')
       
    #===========================================================================



    #===========================================================================
    @utils.timing
    def load_params(self, FilePath):
        """

        Args:
            
            
        """

        if (FilePath[-1] == '/'):
            last = max(os.listdir(FilePath), key=lambda x: int(x.split('.')[0]))
            if last:
                FilePath = FilePath + "/" + last

        print('\n[ROMNet - model_tf.py    ]:   Loading ML Model Parameters from File: ', FilePath)

        self.net.load_weights(FilePath, by_name=True)#, skip_mismatch=False)

    #===========================================================================



    #===========================================================================
    @utils.timing
    def save_params(self, FilePath):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_tf.py    ]:   Saving ML Model Parameters to File: ', FilePath)

        # Save weights
        self.net.save_weights(FilePath, overwrite=True, save_format='h5')

    #===========================================================================



    #===========================================================================
    def predict(self, input_data):
        """

        Args:
            
            
        """

        y_pred = self.net.predict(input_data)
        #y_pred = self.net.call_predict(input_data)

        if (self.norm_output_flg):

            if (not self.got_stats):
                self.read_data_statistics()

            if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
                y_pred = y_pred * self.output_std + self.output_mean

            elif (self.data_preproc_type == '0to1'):
                y_pred = y_pred * self.output_range + self.output_min

            elif (self.data_preproc_type == 'range'):
                y_pred = y_pred * self.output_range

            elif (self.data_preproc_type == '-1to1'):
                y_pred = (y_pred + 1.)/2. * self.output_range + self.output_min

            elif (self.data_preproc_type == 'pareto'):
                y_pred = y_pred * np.sqrt(self.output_std) + self.output_mean

            elif (self.data_preproc_type == 'log'):
                y_pred = np.exp(y_pred) + self.output_min - 1.e-10

            elif (self.data_preproc_type == 'log10'):
                y_pred = 10.**y_pred + self.output_min - 1.e-10


        return y_pred

    #===========================================================================



    #===========================================================================
    def save_data_statistics(self, stat_input, stat_output):

        path = Path( self.path_to_run_fld+'/Data/' )
        path.mkdir(parents=True, exist_ok=True)

        DataNew            = pd.concat([stat_input['mean'], stat_input['std'], stat_input['min'], stat_input['max']], axis=1)
        DataNew.columns    = ['input_mean','input_std','input_min','input_max']
        DataNew.to_csv( self.path_to_run_fld + "/Data/stats_inputs.csv", index=False)  

        DataNew            = pd.concat([stat_output['mean'], stat_output['std'], stat_output['min'], stat_output['max']], axis=1)
        DataNew.columns    = ['output_mean','output_std','output_min','output_max']
        DataNew.to_csv( self.path_to_run_fld + "/Data/stats_output.csv", index=False)

        self.output_mean  = stat_output['mean']
        self.output_std   = stat_output['std']
        self.output_min   = stat_output['min']
        self.output_max   = stat_output['max']
        self.output_range = self.output_max - self.output_min

        self.got_stats = True

    #===========================================================================



    #===========================================================================
    def read_data_statistics(self, PathToRead=None):


        if (PathToRead):
            DataNew = pd.read_csv(PathToRead)
        else:
            DataNew = pd.read_csv(self.path_to_run_fld + "/Data/stats_output.csv")

        self.output_mean  = DataNew['output_mean'].to_numpy()
        self.output_std   = DataNew['output_std'].to_numpy()
        self.output_min   = DataNew['output_min'].to_numpy()
        self.output_max   = DataNew['output_max'].to_numpy()
        self.output_range = self.output_max - self.output_min

        self.stat_output         = {}
        self.stat_output['mean'] = self.output_mean
        self.stat_output['std']  = self.output_std 
        self.stat_output['min']  = self.output_min 
        self.stat_output['max']  = self.output_max 

        self.got_stats = True

    #===========================================================================