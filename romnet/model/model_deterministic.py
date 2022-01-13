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
        print('\n[ROMNet]:   Initializing the ML Model')

        self.SurrogateType     = InputData.SurrogateType
        self.ProbApproach      = InputData.ProbApproach

        self.TrainIntFlg       = InputData.TrainIntFlg
        self.PathToLoadFld     = InputData.PathToLoadFld

        self.got_stats         = False
        try:
            self.ynorm_flg     = InputData.NormalizeOutput
        except:
            self.ynorm_flg     = False

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

            print("\n[ROMNet]:   Trained Model can be Found here: "    + 
                                                              self.PathToRunFld)
            print("\n[ROMNet]:   TensorBoard Data can be Found here: " + 
                                                          self.TBCheckpointFldr)

            # Copying Input File
            print("\n[ROMNet]:   Copying Input File From: " +                  \
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
        print('\n[ROMNet]:   Building the ML Model')

        self.data        = data


        #-----------------------------------------------------------------------
        if (InputData.TransferFlg): 
            self.NN_Transfer_Model = load_weights_(InputData.PathToTransFld)
        else:
            self.NN_Transfer_Model = None

        if (self.TrainIntFlg > 0):
            self.xnorm = self.data.xnorm
        else:
            self.xnorm = None
        #-----------------------------------------------------------------------


        #-----------------------------------------------------------------------
        self.net = Net(InputData, self.xnorm, self.NN_Transfer_Model)

        self.net.AntiPCA_flg = False
        # try:
        #     self.net.AntiPCA_flg = not data.system.ROM_pred_flag
        # else:
        #     self.net.AntiPCA_flg = False

        # if (self.net.AntiPCA_flg):
        #     self.net.A_AntiPCA   = data.system.A
        #     self.net.C_AntiPCA   = data.system.C
        #     self.net.D_AntiPCA   = data.system.D

        self.net.build((1,self.net.NVarsx))
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
        print('\n[ROMNet]:   Compiling the ML Model ... ')


        # #-----------------------------------------------------------------------
        # print('\n[ROMNet]:   Creating the Models Graph')

        # self.graph = self.net.get_graph()

        # self.graph.summary()

        # with open(self.PathToRunFld+'/Model/Model_Summary.txt', 'w') as f:
        #     self.graph.summary(print_fn=lambda x: f.write(x + '\n'))
        
        # tf.keras.utils.plot_model(self.graph, self.PathToRunFld + 
        #                                                      "/Model/Model.png")

        # ModelFile = self.PathToRunFld + '/Model/' + self.net.Name + '/'
        # self.graph.save(ModelFile)

        # #-----------------------------------------------------------------------


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
        if (self.ynorm_flg):
            self.save_data_statistics(self.data.xstat, self.data.ystat)
            self.data.system.ynorm_flg = True
            self.data.system.y_mean    = self.y_mean
            self.data.system.y_std     = self.y_std
            self.data.system.y_min     = self.y_min
            self.data.system.y_max     = self.y_max
            self.data.system.y_range   = self.y_range
        else:
            self.data.system.ynorm_flg = False
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
        print('\n[ROMNet]:   Training the ML Model ... ')

        self.loss_history = LossHistory()
    
        self.NOutputVars  = self.data.NOutputVars
        CallBacksList     = callbacks.get_callback(self, InputData)

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

        print('\n[ROMNet]:   Loading ML Model Parameters from File: ', FilePath)

        self.net.load_weights(FilePath, by_name=True)#, skip_mismatch=False)

    #===========================================================================



    #===========================================================================
    @utils.timing
    def save_params(self, FilePath):
        """

        Args:
            
            
        """
        print('\n[ROMNet]:   Saving ML Model Parameters to File: ', FilePath)

        # Save weights
        self.net.save_weights(FilePath, overwrite=True, save_format='h5')

    #===========================================================================



    #===========================================================================
    def predict(self, xData):
        """

        Args:
            
            
        """

        y_pred = self.net.predict(xData)

        if (self.ynorm_flg):

            if (not self.got_stats):
                self.read_data_statistics()

            y_pred = y_pred * self.y_range + self.y_min

        return y_pred

    #===========================================================================



    #===========================================================================
    def save_data_statistics(self, xstat, ystat):

        path = Path( self.PathToRunFld+'/Data/' )
        path.mkdir(parents=True, exist_ok=True)

        DataNew            = pd.concat([xstat['mean'], xstat['std'], xstat['min'], xstat['max']], axis=1)
        DataNew.columns    = ['x_mean','x_std','x_min','x_max']
        DataNew.to_csv( self.PathToRunFld + "/Data/x_stats.csv", index=False)  

        DataNew            = pd.concat([ystat['mean'], ystat['std'], ystat['min'], ystat['max']], axis=1)
        DataNew.columns    = ['y_mean','y_std','y_min','y_max']
        DataNew.to_csv( self.PathToRunFld + "/Data/y_stats.csv", index=False)

        self.y_mean  = ystat['mean']
        self.y_std   = ystat['std']
        self.y_min   = ystat['min']
        self.y_max   = ystat['max']
        self.y_range = self.y_max - self.y_min

        self.got_stats = True

    #===========================================================================



    #===========================================================================
    def read_data_statistics(self, PathToRead=None):


        if (PathToRead):
            DataNew = pd.read_csv(PathToRead)
        else:
            DataNew = pd.read_csv(self.PathToRunFld + "/Data/y_stats.csv")

        self.y_mean  = DataNew['y_mean'].to_numpy()
        self.y_std   = DataNew['y_std'].to_numpy()
        self.y_min   = DataNew['y_min'].to_numpy()
        self.y_max   = DataNew['y_max'].to_numpy()
        self.y_range = self.y_max - self.y_min

        self.got_stats = True

    #===========================================================================