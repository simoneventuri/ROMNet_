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

        self.Type                = InputData.data_type

        self.path_to_data_fld    = InputData.path_to_data_fld
        self.path_to_load_fld    = InputData.path_to_load_fld
        self.InputFiles          = InputData.InputFiles
        self.OutputFiles         = InputData.OutputFiles
        
        self.input_vars          = InputData.input_vars_all
        self.n_inputs            = len(self.input_vars)

        try:    
            self.output_vars     = system.output_vars
        except:
            self.output_vars     = InputData.output_vars
        self.n_outputs           = len(self.output_vars)

        try:
            self.trans_fun       = InputData.trans_fun
        except:
            self.trans_fun       = None

        try:
            self.norm_output_flg = InputData.norm_output_flg
        except:
            self.norm_output_flg = False

        try:
            self.valid_perc      = InputData.valid_perc
        except:
            self.valid_perc      = 0.
        try:
            self.test_perc       = InputData.TestPerc
        except:
            self.test_perc       = 0.

        self.surrogate_type      = InputData.surrogate_type

        self.NData               = 0
        self.xtrain, self.ytrain = None, None
        self.xtest,  self.ytest  = None, None

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
        print('[ROMNet - blackbox.py]:   Reading Data')

        self.n_train_tot         = {}  
        self.train               = {}
        self.valid               = {}
        self.all                 = {}
        self.test                = {}
        self.extra               = {}
        FirstFlg                 = True
        for data_id, InputFile in self.InputFiles.items():

            if isinstance(self.input_vars, (list,tuple)):
                input_vars = self.input_vars
            else:
                input_vars = list(pd.read_csv(self.path_to_data_fld+'/train/'+data_id+'/'+self.input_vars[data_id], header=None).to_numpy()[0,:])

            if isinstance(self.output_vars, (list,tuple)):
                output_vars = self.output_vars
            else:
                output_vars = list(pd.read_csv(self.path_to_data_fld+'/train/'+data_id+'/'+self.output_vars[data_id], header=None).to_numpy()[0,:])

            Data = pd.read_csv(self.path_to_data_fld+'/train/'+data_id+'/'+InputFile, header=0)


            if (self.surrogate_type == 'DeepONet'):

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
                xall = Data[input_vars]           

            for iCol in range(xall.shape[1]):
                array_sum = np.sum(xall.to_numpy()[:,iCol])
                if (np.isnan(array_sum)):
                    print('[ROMNet -  blackbox.py              ]:   xall has NaN!!!')


            Data = pd.read_csv(self.path_to_data_fld+'/train/'+data_id+'/'+self.OutputFiles[data_id], header=0)
            yall = Data[output_vars]
            for iCol in range(yall.shape[1]):
                array_sum = np.sum(yall.to_numpy()[:,iCol])
                if (np.isnan(array_sum)):
                    print('[ROMNet -  blackbox.py              ]:   yall has NaN!!!')


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
                self.norm_input      = xall
                if (data_id != 'res'):
                    self.norm_output = yall
            else:
                self.norm_input      = self.norm_input.append(xall, ignore_index=True)
                if (data_id != 'res'):
                    self.norm_output = self.norm_output.append(yall, ignore_index=True)
            FirstFlg = False
        
            self.train[data_id] = [xtrain, ytrain]
            self.valid[data_id] = [xvalid, yvalid]
            self.test[data_id]  = [xtest,   ytest]
            self.all[data_id]   = [xall,     yall]
            self.extra[data_id] = []

        self.transform_normalization_data()
        self.compute_input_statistics()      
        self.compute_output_statistics()      

        if (self.norm_output_flg):
            if (self.path_to_load_fld):
                self.read_output_statistics(self.path_to_load_fld)      
            self.train, self.valid = self.normalize_output_data([self.train, self.valid])

        self.train, self.valid = self.system.preprocess_data([self.train, self.valid], self.xstat)

        print("[ROMNet -  blackbox.py              ]:   Train      Data: ", self.train)
        print("[ROMNet -  blackbox.py              ]:   Validation Data: ", self.valid)

    #===========================================================================



    #===========================================================================
    def get_num_pts(self, data_type='training', verbose=1):

        def print_fn(dset, data_type, verbose):
            num_pts = {data_id: dset.n_samples[data_id] for data_id in dset.data}
            if verbose:
                print("[ROMNet -  blackbox.py              ]: Number of pts for data " + k + ": ", v)
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

