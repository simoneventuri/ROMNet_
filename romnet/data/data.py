import numpy      as np
import pandas     as pd
from pathlib  import Path


class Data(object):

    def __init__(self, InputData, system):

        self.Type = InputData.DataType


    # ---------------------------------------------------------------------------------------------------------------------------
    def transform_normalization_data(self):

        if (self.trans_fun):
            for ifun, fun in enumerate(self.trans_fun):
                vars_list = self.trans_fun[fun]

                for ivar, var in enumerate(self.input_vars):
                    if var in vars_list:
                        if (fun == 'log10'):
                            self.norm_input[var] = np.log10(self.norm_input[var])
                        elif (fun == 'log'):
                            self.norm_input[var] = np.log(self.norm_input[var])

                for ivar, var in enumerate(self.output_vars):
                    if var in vars_list:
                        if (fun == 'log10'):
                            self.norm_output[var] = np.log10(self.norm_output[var])
                        elif (fun == 'log'):
                            self.norm_output[var] = np.log(self.norm_output[var])
                    
    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def normalize_input_data(self, all_data):

        for i, now_data in enumerate(all_data):
            for data_id, xyi_data in now_data.items():
                if (data_id != 'res'):
                    x_data = xyi_data[0]

                    all_data[i][data_id][0] = (x_data - self.stat_input['min'].to_numpy()) / (self.stat_input['max'].to_numpy() - self.stat_input['min'].to_numpy())
                    #all_data[i][data_id][1] = (x_data - self.system.C) / self.system.D
        
        return all_data

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def normalize_output_data(self, all_data):

        for i, now_data in enumerate(all_data):
            for data_id, xyi_data in now_data.items():
                if (data_id != 'res'):
                    y_data = xyi_data[1]

                    all_data[i][data_id][1] = (y_data - self.stat_output['min'].to_numpy()) / (self.stat_output['max'].to_numpy() - self.stat_output['min'].to_numpy())
                    #all_data[i][data_id][1] = (y_data - self.system.C) / self.system.D
        
        return all_data

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def compute_input_statistics(self):

        self.stat_input         = {}
        self.stat_input['min']  = self.norm_input.min(axis = 0)
        self.stat_input['max']  = self.norm_input.max(axis = 0)
        self.stat_input['mean'] = self.norm_input.mean(axis = 0)
        self.stat_input['std']  = self.norm_input.std(axis = 0)   

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def compute_output_statistics(self):

        self.stat_output         = {}
        self.stat_output['min']  = self.norm_output.min(axis = 0)
        self.stat_output['max']  = self.norm_output.max(axis = 0)
        self.stat_output['mean'] = self.norm_output.mean(axis = 0)
        self.stat_output['std']  = self.norm_output.std(axis = 0)   

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def read_output_statistics(self, PathToRead=None):

        if (PathToRead):
            DataNew = pd.read_csv(PathToRead + "/Data/stats_output.csv")
        else:
            DataNew = pd.read_csv(self.PathToRunFld + "/Data/stats_output.csv")

        self.stat_output         = {}
        self.stat_output['mean'] = DataNew['output_mean']
        self.stat_output['std']  = DataNew['output_std']
        self.stat_output['min']  = DataNew['output_min']
        self.stat_output['max']  = DataNew['output_max']

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def res_fn(self, net):
        '''Residual loss function'''

        self.n_inputs  = net.n_inputs
        self.n_outputs = net.n_outputs

        def residual(inputs, training=True):
            return None
        
        return residual

    # ---------------------------------------------------------------------------------------------------------------------------
