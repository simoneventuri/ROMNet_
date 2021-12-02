import numpy as np

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
                            self.xnorm[var] = np.log10(self.xnorm[var] + 1.e-15)

                for ivar, var in enumerate(self.OutputVars):
                    if var in vars_list:
                        if (fun == 'log10'):
                            self.ynorm[var] = np.log10(self.xnorm[var] + 1.e-15)
                    
    #===========================================================================



    #===========================================================================
    def compute_statistics(self):

        self.xstat         = {}
        self.xstat['min']  = self.xnorm.min(axis = 0)
        self.xstat['max']  = self.xnorm.max(axis = 0)
        self.xstat['mean'] = self.xnorm.mean(axis = 0)
        self.xstat['std']  = self.xnorm.std(axis = 0)   

        self.ystat         = {}
        self.ystat['min']  = self.ynorm.min(axis = 0)
        self.ystat['max']  = self.ynorm.max(axis = 0)
        self.ystat['mean'] = self.ynorm.mean(axis = 0)
        self.ystat['std']  = self.ynorm.std(axis = 0)   

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
