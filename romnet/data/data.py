import numpy      as np
import pandas     as pd
from pathlib  import Path


class Data(object):

    def __init__(self, InputData, system):

        self.Type = InputData.DataType


    #===========================================================================
    def transform_normalization_data(self):

        if (self.TransFun):
            for ifun, fun in enumerate(self.TransFun):
                vars_list = self.TransFun[fun]

                for ivar, var in enumerate(self.InputVars):
                    if var in vars_list:
                        if (fun == 'log10'):
                            #self.xnorm[var] = np.log10(self.xnorm[var] + 1.e-15)
                            self.xnorm[var] = np.log(self.xnorm[var] + 1.e-15)

                for ivar, var in enumerate(self.OutputVars):
                    if var in vars_list:
                        if (fun == 'log10'):
                            #self.ynorm[var] = np.log10(self.xnorm[var] + 1.e-15)
                            self.ynorm[var] = np.log(self.xnorm[var] + 1.e-15)
                    
    #===========================================================================


    #===========================================================================
    def normalize_output_data(self, all_data):

        for i, now_data in enumerate(all_data):
            for data_id, xyi_data in now_data.items():
                if (data_id != 'res'):
                    y_data = xyi_data[1]

                    all_data[i][data_id][1] = (y_data - self.ystat['min'].to_numpy()) / (self.ystat['max'].to_numpy() - self.ystat['min'].to_numpy())
                    #all_data[i][data_id][1] = (y_data - self.system.C) / self.system.D
        
        return all_data

    #===========================================================================


    #===========================================================================
    def compute_input_statistics(self):

        self.xstat         = {}
        self.xstat['min']  = self.xnorm.min(axis = 0)
        self.xstat['max']  = self.xnorm.max(axis = 0)
        self.xstat['mean'] = self.xnorm.mean(axis = 0)
        self.xstat['std']  = self.xnorm.std(axis = 0)   

    #===========================================================================



    #===========================================================================
    def compute_output_statistics(self):

        self.ystat         = {}
        self.ystat['min']  = self.ynorm.min(axis = 0)
        self.ystat['max']  = self.ynorm.max(axis = 0)
        self.ystat['mean'] = self.ynorm.mean(axis = 0)
        self.ystat['std']  = self.ynorm.std(axis = 0)   

    #===========================================================================



    #===========================================================================
    def read_output_statistics(self, PathToRead=None):

        if (PathToRead):
            DataNew = pd.read_csv(PathToRead + "/Data/y_stats.csv")
        else:
            DataNew = pd.read_csv(self.PathToRunFld + "/Data/y_stats.csv")

        self.ystat         = {}
        self.ystat['mean'] = DataNew['y_mean']
        self.ystat['std']  = DataNew['y_std']
        self.ystat['min']  = DataNew['y_min']
        self.ystat['max']  = DataNew['y_max']

    #===========================================================================



    #===========================================================================
    def res_fn(self, net):
        '''Residual loss function'''

        self.NVarsx = net.NVarsx
        self.NVarsy = net.NVarsy

        def residual(inputs, training=True):
            return None
        
        return residual
    #===========================================================================
