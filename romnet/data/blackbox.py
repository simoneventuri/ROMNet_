import pandas as pd
import numpy  as np
import abc
import os 

from .data   import Data
from ..utils import run_if_any_none



class BlackBox(Data):

    #===========================================================================
    def __init__(self, InputData, system):
        super(BlackBox, self).__init__(InputData, system)

        self.Type                = InputData.DataType

        self.PathToDataFld       = InputData.PathToDataFld
        self.PathToLoadFld       = InputData.PathToLoadFld
        self.InputFiles          = InputData.InputFiles
        self.OutputFiles         = InputData.OutputFiles

        self.valid_perc          = InputData.ValidPerc
        self.test_perc           = InputData.TestPerc

        self.SurrogateType       = InputData.SurrogateType
        if (self.SurrogateType == 'DeepONet'):
            self.BranchVars      = InputData.BranchVars
            self.TrunkVars       = InputData.TrunkVars
        else:
            self.InputVars       = InputData.InputVars

        try:    
            self.OutputVars      = system.OutputVars
        except:    
            self.OutputVars      = InputData.OutputVars
        self.NOutputVars         = len(self.OutputVars)
    
        self.NData               = 0
        self.xtrain, self.ytrain = None, None
        self.xtest,  self.ytest  = None, None

        try:
            self.TransFun        = InputData.TransFun
        except:
            self.TransFun        = None

        try:
            self.ynorm_flg     = InputData.NormalizeOutput
        except:
            self.ynorm_flg     = False

        self.system              = system
        self.other_idxs          = None
        self.ind_idxs            = None
        self.size_splits         = None
        self.order               = None
        self.get_residual        = None
        self.grad_fn             = None
        self.fROM_anti           = None
    
    #===========================================================================


    #===========================================================================
    # Reading Data 
    def get(self, InputData):
        print('[ROMNet]:   Reading Data')

        self.n_train_tot         = {}  
        self.train               = {}
        self.valid               = {}
        self.all                 = {}
        self.test                = {}
        self.extra               = {}
        FirstFlg                 = True
        for data_id, InputFile in self.InputFiles.items():

            if isinstance(self.InputVars, (list,tuple)):
                InputVars = self.InputVars
            else:
                InputVars = list(pd.read_csv(self.PathToDataFld+'/train/'+data_id+'/'+self.InputVars[data_id], header=None).to_numpy()[0,:])

            if isinstance(self.OutputVars, (list,tuple)):
                OutputVars = self.OutputVars
            else:
                OutputVars = list(pd.read_csv(self.PathToDataFld+'/train/'+data_id+'/'+self.OutputVars[data_id], header=None).to_numpy()[0,:])

            Data = pd.read_csv(self.PathToDataFld+'/train/'+data_id+'/'+InputFile, header=0)


            if (self.SurrogateType == 'DeepONet'):

                uall = Data[self.BranchVars]
                if (len(self.TrunkVars) > 0):
                    tall         = Data[self.TrunkVars]
                    xall         = pd.concat([uall, tall], axis=1)
                else:
                    xall         = uall

                # if (not self.BranchScale == None):
                #     for Var in self.BranchVars:
                #         xall[Var] = xall[Var].apply(lambda x: 
                #                                          self.BranchScale(x+1.e-15))
                # if (not self.TrunkScale == None):
                #     for Var in self.TrunkVars:
                #         xall[Var] = xall[Var].apply(lambda x: 
                #                                           self.TrunkScale(x+1.e-15))

            else:
                xall = Data[InputVars]           

            for iCol in range(xall.shape[1]):
                array_sum = np.sum(xall.to_numpy()[:,iCol])
                if (np.isnan(array_sum)):
                    print('xall has NaN!!!')


            Data = pd.read_csv(self.PathToDataFld+'/train/'+data_id+'/'+self.OutputFiles[data_id], header=0)
            yall = Data[OutputVars]
            for iCol in range(yall.shape[1]):
                array_sum = np.sum(yall.to_numpy()[:,iCol])
                if (np.isnan(array_sum)):
                    print('yall has NaN!!!')


            xtrain     = xall.copy()
            xtest      = xall.copy()
            xtrain     = xtrain.sample(frac=(1.0-self.test_perc/100.0), random_state=3)
            self.n_train_tot[data_id] = len(xtrain)
            xtest      = xtest.drop(xtrain.index)
            xvalid     = xtrain.copy()
            xtrain     = xtrain.sample(frac=(1.0-self.valid_perc/100.0), random_state=3)
            xvalid     = xvalid.drop(xtrain.index)

            ytrain     = yall.copy()
            ytest      = yall.copy()
            ytrain     = ytrain.sample(frac=(1.0-self.test_perc/100.0), random_state=3)
            ytest      = ytest.drop(ytrain.index)
            yvalid     = ytrain.copy()
            ytrain     = ytrain.sample(frac=(1.0-self.valid_perc/100.0), random_state=3)
            yvalid     = yvalid.drop(ytrain.index)


            if (FirstFlg):
                self.xnorm     = xall
                if (data_id != 'res'):
                    self.ynorm = yall
            else:
                self.xnorm     = self.xnorm.append(xall, ignore_index=True)
                if (data_id != 'res'):
                    self.ynorm = self.ynorm.append(yall, ignore_index=True)
            FirstFlg = False
        
            self.train[data_id] = [xtrain, ytrain]
            self.valid[data_id] = [xvalid, yvalid]
            self.test[data_id]  = [xtest,   ytest]
            self.all[data_id]   = [xall,     yall]
            self.extra[data_id] = []

        self.transform_normalization_data()
        self.compute_input_statistics()      
        self.compute_output_statistics()      

        if (self.ynorm_flg):
            if (self.PathToLoadFld):
                self.read_output_statistics(self.PathToLoadFld)      
            self.train, self.valid = self.normalize_output_data([self.train, self.valid])

        self.train, self.valid = self.system.preprocess_data([self.train, self.valid], self.xstat)

        print("[ROMNet]:   Train      Data: ", self.train)
        print("[ROMNet]:   Validation Data: ", self.valid)

    #===========================================================================



    #===========================================================================
    def get_num_pts(self, data_type='training', verbose=1):

        def print_fn(dset, data_type, verbose):
            num_pts = {data_id: dset.n_samples[data_id] for data_id in dset.data}
            if verbose:
                print("Number of pts for data " + k + ": ", v)
                for k, v in num_pts.items():
                    utils.print_submain("  - '%s': %8d" % (k, v))
            return num_pts

        super(BlackBox, self).get_num_pts( data_type=data_type, verbose=verbose, print_fn=print_fn )

    #===========================================================================



    #===========================================================================
    def get_train_valid(self, data):

        train, valid = {}, {}
        for i, data_i in data:
            train[i], valid[i] = super(BlackBox, self).get_train_valid(data_i)

        return train, valid

    #===========================================================================

