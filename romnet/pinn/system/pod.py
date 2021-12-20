import numpy                    as np
from scipy.integrate import ode as scipy_ode
import pandas                   as pd
import tensorflow               as tf

from .system import System



class POD(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):
        ROMNetFldr        = InputData.ROMNetFldr

        # # Integrator time step
        # self.dt0          = 1.e-3
        # self.dt_str       = 1.
        # self.dt_max       = 0.02

        # # Integration method
        # self.method       = 'bdf'


        # self.order        = [1]

        # self.ind_names    = ['t']
        # self.other_names  = ['x','v']

        # self.ind_labels   = ['t [s]']
        # self.other_labels = ['x [m]','v [m/s]']

        # self.get_variable_locations()


        # # Initial/final time
        # self.read_extremes(ROMNetFldr)
        
        # # Get Parameters
        # self.read_params(ROMNetFldr)
        
        # # Get ode matrices
        # self.K           = self.get_matrix()

        # self.fROM_anti   = None
        
    #===========================================================================



    #===========================================================================
    def f(self, t, y, arg):
        return None

    #===========================================================================



    #===========================================================================
    def jac(self, t, y, arg):
        return None

    #===========================================================================
