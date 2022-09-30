import numpy                    as np
from scipy.integrate        import solve_ivp
import pandas                   as pd
import tensorflow               as tf

from .system import System



class TransTanh(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):
        ROMNet_fld        = InputData.ROMNet_fld

        # Integrator time step
        self.dt0          = 1.e-3
        self.dt_str       = 1.
        self.dt_max       = 0.02

        # Integration method
        self.method       = 'bdf'


        self.order        = [1]

        self.ind_names    = ['t']
        self.other_names  = ['x']

        self.ind_labels   = ['t [s]']
        self.other_labels = ['x [m]']

        self.get_variable_locations()

        # Get Parameters
        self.read_params(ROMNet_fld)

        self.fROM_anti    = None

        self.ind_ranges     = {}     
        self.other_ranges   = {}  

    #===========================================================================



    #===========================================================================
    def f(t, params):
        return [1./(np.cosh(params[0]/params[1]-t)**2)]

    #===========================================================================



    #===========================================================================
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''
        Params  = y_0
        tout    = np.linspace(0.,15.,Nt)

        output  = solve_ivp(self.f, tout[[0,-1]], y_0, method='BDF', t_eval=tout, rtol=1.e-15, atol=1.e-20 )
    
        # Appending Data
        t = tout[...,np.newaxis]
        y = output.y.T 

        return [t, y]

    #===========================================================================



    #===========================================================================
    def get_residual(self, ic, t, grads):

        y, dy_dt = grads

        #tf.print('ic = ', ic)
        return dy_dt - 1./tf.math.cosh(ic/self.params[0]-t)**2

    #===========================================================================



    #===========================================================================
    def read_extremes(self, ROMNet_fld):

        PathToExtremes = ROMNet_fld + '/database/TransTanh/Extremes/'

        Data                = pd.read_csv(PathToExtremes+'/t.csv')
        self.t0             = Data.to_numpy()[0,:]
        self.tEnd           = Data.to_numpy()[1,:]

        self.ind_extremes   = {'t': [self.t0, self.tEnd]}


        Data                = pd.read_csv(PathToExtremes+'/x.csv')
        self.xMin           = Data.to_numpy()[0,:]
        self.xMax           = Data.to_numpy()[1,:]

        self.other_extremes = {'x': [self.xMin[0], self.xMax[0]]}

        self.from_extremes_to_ranges()

    #===========================================================================


    #===========================================================================
    def read_params(self, ROMNet_fld):

        PathToParams = ROMNet_fld + '/database/TransTanh/Params/'

        a = pd.read_csv(PathToParams+'/a.csv').to_numpy()[0,0]

        self.params = [a]

    #===========================================================================