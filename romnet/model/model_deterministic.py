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

from .model        import Model
from ..training    import LossHistory, get_loss, get_optimizer, callbacks
from ..nn          import load_model_, load_weights_
from ..            import utils
from ..metrics     import get_metric



#===============================================================================
from datetime import datetime

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return get_curr_time()

#===============================================================================



#===============================================================================
class Model_Deterministic(Model):     
    """

    Args:
        
        
    """



    #===========================================================================
    # Class Initialization
    def __init__(self, InputData):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_deterministic.py    ]:   Initializing the ML Model')

        self.SurrogateType       = InputData.SurrogateType
        self.ProbApproach        = InputData.ProbApproach
  
        self.TrainIntFlg         = InputData.TrainIntFlg
        self.PathToLoadFld       = InputData.PathToLoadFld
  
        self.got_stats           = False
        try:
            self.norm_output_flg = InputData.norm_output_flg
        except:
            self.norm_output_flg = False

        if (self.TrainIntFlg > 0):

            #-------------------------------------------------------------------
            ### Dealing With Paths

            InputData.PathToRunFld += '/' + self.SurrogateType + '/' + self.ProbApproach + '/'
            path = Path(InputData.PathToRunFld)
            path.mkdir(parents=True, exist_ok=True)

            Prefix = 'Run_'
            if (InputData.NNRunIdx == 0):
                if (len([x for x in os.listdir(InputData.PathToRunFld) 
                                                          if 'Run_' in x]) > 0):
                    InputData.NNRunIdx = str(np.amax( np.array( [int(x[len(Prefix):]) for x in os.listdir(InputData.PathToRunFld) if Prefix in x], dtype=int) ) + 1)
                else:
                    InputData.NNRunIdx = 1

            InputData.TBCheckpointFldr = InputData.PathToRunFld + '/TB/' +     \
               Prefix + str(InputData.NNRunIdx) + "_{}".format(get_start_time())

            InputData.PathToRunFld    += '/' + Prefix + str(InputData.NNRunIdx)

            self.PathToRunFld      = InputData.PathToRunFld
            self.TBCheckpointFldr  = InputData.TBCheckpointFldr

            # Creating Folders
            path = Path( self.PathToRunFld+'/Figures/' )
            path.mkdir(parents=True, exist_ok=True)

            path = Path( self.PathToRunFld+'/Model/Params' )
            path.mkdir(parents=True, exist_ok=True)

            path = Path( self.PathToRunFld+'/Training/Params' )
            path.mkdir(parents=True, exist_ok=True)
                     
            print("\n[ROMNet - model_deterministic.py    ]:   Trained Model can be Found here: "    + 
                                                              self.PathToRunFld)
            print("\n[ROMNet - model_deterministic.py    ]:   TensorBoard Data can be Found here: " + 
                                                          self.TBCheckpointFldr)

            # Copying Input File
            print("\n[ROMNet - model_deterministic.py    ]:   Copying Input File From: " + \
                              InputData.InputFilePath+'/ROMNet_Input.py to ' + \
                                         self.PathToRunFld + '/ROMNet_Input.py')
            shutil.copyfile(InputData.InputFilePath+'/ROMNet_Input.py', 
                                         self.PathToRunFld + '/ROMNet_Input.py')

            #-------------------------------------------------------------------

        else:

            self.PathToRunFld = InputData.PathToRunFld

    #===========================================================================



    #===========================================================================
    @utils.timing
    def build(
        self,
        InputData,
        data,
        Net,
        loadfile_no=None,
    ):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_deterministic.py    ]:   Building the ML Model')

        self.data        = data


        #-----------------------------------------------------------------------
        if (InputData.TransferFlg): 
            self.NN_Transfer_Model = load_weights_(InputData.PathToTransFld)
        else:
            self.NN_Transfer_Model = None

        if (self.TrainIntFlg > 0):
            self.norm_input  = self.data.norm_input
            self.stat_output = self.data.stat_output
        else:
            try:
                self.read_data_statistics()
            except:
                self.stat_output = None
            self.norm_input  = None
            
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.net = Net(InputData, self.norm_input, self.stat_output)

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
        if (self.TrainIntFlg == 0):
            if (loadfile_no):
                self.load_params(self.PathToRunFld + "/Training/Params/"+loadfile_no+".h5")
            else:
                self.load_params(self.PathToRunFld + "/Training/Params/")
        else:
            if (self.PathToLoadFld is not None):
                if (loadfile_no):
                    self.load_params(self.PathToLoadFld + "/Training/Params/"+loadfile_no+".h5")
                if (self.PathToLoadFld[-1] == '/'):
                    self.load_params(self.PathToLoadFld + "/Training/Params/")
                else:
                    self.load_params(self.PathToLoadFld)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.save_params(self.PathToRunFld+'/Model/Params/Initial.h5')
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
        print('\n[ROMNet - model_deterministic.py    ]:   Compiling the ML Model ... ')


        #-----------------------------------------------------------------------
        print('\n[ROMNet - model_deterministic.py    ]:   Creating the Models Graph')

        self.graph = self.net.get_graph()

        self.graph.summary()

        with open(self.PathToRunFld+'/Model/Model_Summary.txt', 'w') as f:
            self.graph.summary(print_fn=lambda x: f.write(x + '\n'))
        
        tf.keras.utils.plot_model(self.graph, self.PathToRunFld + 
                                                             "/Model/Model.png")

        ModelFile = self.PathToRunFld + '/Model/' + self.net.structure_name + '/'
        self.graph.save(ModelFile)

        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        opt = get_optimizer(InputData)
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if (InputData.LossWeights is not None):
            self.loss_weights = {}
            for data_id in InputData.LossWeights.keys():
                self.loss_weights[data_id] =                                   \
                                 list(InputData.LossWeights[data_id].values()) \
                           if isinstance(InputData.LossWeights[data_id], dict) \
                                             else InputData.LossWeights[data_id]
        else:
            self.loss_weights = None
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        if (self.norm_output_flg):
            self.save_data_statistics(self.data.stat_input, self.data.stat_output)
            self.data.system.norm_output_flg = True
            self.data.system.output_mean    = self.output_mean
            self.data.system.output_std     = self.output_std
            self.data.system.output_min     = self.output_min
            self.data.system.output_max     = self.output_max
            self.data.system.output_range   = self.output_range
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
        for data_id, LossID in InputData.Losses.items():
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
        if InputData.Metrics is not None:
            self.metrics = [ get_metric(met) for met in InputData.Metrics ]
        else:
            self.metrics = None
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        try:
            run_eagerly_flg = InputData.RunEagerlyFlg
        except:
            run_eagerly_flg = False

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
        print('\n[ROMNet - model_deterministic.py    ]:   Training the ML Model ... ')

        self.loss_history = LossHistory()
    
        self.n_outputs = self.data.n_outputs
        CallBacksList  = callbacks.get_callback(self, InputData)

        if (InputData.DataType == 'PDE'):
            x        = self.data.train
            y        = None
            xy_valid = self.data.valid 
        else:
            x        = self.data.train['ext'][0]
            y        = self.data.train['ext'][1]
            xy_valid = (self.data.valid['ext'][0], self.data.valid['ext'][1])


        History       = self.net.fit(x=x, 
                                     y=y, 
                                     batch_size=InputData.BatchSize, 
                                     validation_data=xy_valid, 
                                     verbose=1, 
                                     epochs=InputData.NEpoch, 
                                     callbacks=CallBacksList)

        pd.DataFrame.from_dict(History.history).to_csv(
             path_or_buf=self.PathToRunFld+'/Training/History.csv', 
             index=False
             )
        
        self.loss_history.history = History.history


        self.save_params(self.PathToRunFld+'/Model/Params/Final.h5')
       
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

        print('\n[ROMNet - model_deterministic.py    ]:   Loading ML Model Parameters from File: ', FilePath)

        self.net.load_weights(FilePath, by_name=True)#, skip_mismatch=False)

    #===========================================================================



    #===========================================================================
    @utils.timing
    def save_params(self, FilePath):
        """

        Args:
            
            
        """
        print('\n[ROMNet - model_deterministic.py    ]:   Saving ML Model Parameters to File: ', FilePath)

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

            y_pred = y_pred * self.output_range + self.output_min

        return y_pred

    #===========================================================================



    #===========================================================================
    def predict_test(self, input_data):
        """

        Args:
            
            
        """

        y_pred = self.net.call_predict_deeponet_1(input_data)

        # if (self.norm_output_flg):

        #     if (not self.got_stats):
        #         self.read_data_statistics()

        #     y_pred = y_pred * self.output_range + self.output_min

        return y_pred

    #===========================================================================




    #===========================================================================
    def predict_test_2(self, input_data):
        """

        Args:
            
            
        """

        y_pred = self.net.call_predict_deeponet_2(input_data)

        # if (self.norm_output_flg):

        #     if (not self.got_stats):
        #         self.read_data_statistics()

        #     y_pred = y_pred * self.output_range + self.output_min

        return y_pred

    #===========================================================================



    #===========================================================================
    def save_data_statistics(self, stat_input, stat_output):

        path = Path( self.PathToRunFld+'/Data/' )
        path.mkdir(parents=True, exist_ok=True)

        DataNew            = pd.concat([stat_input['mean'], stat_input['std'], stat_input['min'], stat_input['max']], axis=1)
        DataNew.columns    = ['input_mean','input_std','input_min','input_max']
        DataNew.to_csv( self.PathToRunFld + "/Data/stats_inputs.csv", index=False)  

        DataNew            = pd.concat([stat_output['mean'], stat_output['std'], stat_output['min'], stat_output['max']], axis=1)
        DataNew.columns    = ['output_mean','output_std','output_min','output_max']
        DataNew.to_csv( self.PathToRunFld + "/Data/stats_output.csv", index=False)

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
            DataNew = pd.read_csv(self.PathToRunFld + "/Data/stats_output.csv")

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