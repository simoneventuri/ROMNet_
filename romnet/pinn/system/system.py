import pandas as pd
import numpy  as np
import abc



class System(object):

    #===========================================================================
    def __init__(
        self,
        ROMNet_fld,
    ):
        ROMNet_fld          = InputData.ROMNet_fld

        self.order          = 0

        self.ind_extremes   = {}
        self.other_extremes = {}
        
        self.ind_ranges     = {}     
        self.other_ranges   = {}  
        
    #===========================================================================



    #===========================================================================
    def from_extremes_to_ranges(self):
        
        self.ind_ranges     = {}     
        self.other_ranges   = {}  

        for ind_name in self.ind_names:
            self.ind_ranges[ind_name] = np.array([self.ind_extremes[ind_name][0], self.ind_extremes[ind_name][1]], dtype=np.float64)
            
        for other_name in self.other_names:
            self.other_ranges[other_name] = np.array([self.other_extremes[other_name][0], self.other_extremes[other_name][1]], dtype=np.float64)

    #===========================================================================



    #===========================================================================
    def get_variable_locations(self):

        self.names       = self.ind_names  + self.other_names
        self.labels      = self.ind_labels + self.other_labels

        self.n_ind       = len(self.ind_names)
        self.n_other     = len(self.other_names)

        self.ind_idxs    = [self.names.index(self.ind_names[i])   for i in range(self.n_ind)]
        self.other_idxs  = [self.names.index(self.other_names[i]) for i in range(self.n_other)]

    #===========================================================================



    #===========================================================================
    @abc.abstractmethod
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the PDE'''
        ''' '''
    #===========================================================================



    #===========================================================================
    @abc.abstractmethod
    def get_residual(self, ic, t, grads):
        ''' '''
    #===========================================================================
