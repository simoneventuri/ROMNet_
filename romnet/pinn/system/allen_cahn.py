import numpy                    as np
from scipy.integrate import ode as scipy_ode
import pandas                   as pd
import tensorflow               as tf

from .system import System



class Allen_Cahn(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):
        ROMNet_fld        = InputData.ROMNet_fld

        self.order        = [1,2]

        self.ind_names    = ['t', 'x']
        self.other_names  = ['u0_par']

        self.ind_labels   = ['t [s]', 'x [m]']
        self.other_labels = ['u0_par']

        self.get_variable_locations()


        # Initial/final time
        self.read_extremes(ROMNet_fld)
        
        # Get Parameters
        self.read_params(ROMNet_fld)
        
        # Get No of Steps
        self.read_steps(ROMNet_fld)

    #===========================================================================



    #===========================================================================
    def set_initial_condition(self, x, u0_par):

        u0   = x**2 * np.cos(x*np.pi) * u0_par

        return u0

    #===========================================================================



    #===========================================================================
    def f(self, x, u):

        def laplacian_interval( u, x ):

            nx = len( u )

            uxl = ( u[1:nx-1] - u[0:nx-2] ) / ( x[1:nx-1] - x[0:nx-2] )
            uxr = ( u[2:nx]   - u[1:nx-1] ) / ( x[2:nx]   - x[1:nx-1] )

            uxx = 2.0 * ( uxr[0:nx-2] - uxl[0:nx-2] ) / ( x[2:nx] - x[0:nx-2] )
            uxx = np.insert( uxx, 0, 0.0 )
            uxx = np.insert( uxx, nx-1, 0.0 )

            return uxx

        uxx = laplacian_interval( u, x )

        return self.params[0] * uxx - u * ( u**2 - 1.0 ) / ( 2.0 * self.params[1]**2 )

    #===========================================================================



    #===========================================================================
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''

        x          = np.linspace( self.xMin, self.xMax, self.n_x )
        t          = np.linspace( self.t0, self.tEnd, self.n_t )
        dt         = ( self.tEnd - self.t0 ) / ( self.n_t - 1 )
        xMat, tMat = np.meshgrid(x,t)

        uList = []
        u     = self.set_initial_condition( x, y_0 )
        uList.append(u)
        for i in range(1,self.n_t):
            dudt  = self.f( x, u )
            u    += dudt * dt
            uList.append(u)
        uMat  = np.stack(uList)

        return [tMat.flatten(), xMat.flatten(), uMat.flatten()]

    #===========================================================================



    #===========================================================================
    def get_residual(self, ic, t, grads):

        y, dy_dt, dy_dxx = grads

        return dy_dt - ( self.params[0] * dy_dxx - y * ( y**2 - 1.0 ) / ( 2.0 * self.params[1]**2 ) )

    #===========================================================================



    #===========================================================================
    def read_extremes(self, ROMNet_fld):

        PathToExtremes = ROMNet_fld + '/database/Allen_Cahn/Extremes/'

        Data                = pd.read_csv(PathToExtremes+'/t.csv')
        self.t0             = Data.to_numpy()[0,:]
        self.tEnd           = Data.to_numpy()[1,:]

        Data                = pd.read_csv(PathToExtremes+'/x.csv')
        self.xMin           = Data.to_numpy()[0,:]
        self.xMax           = Data.to_numpy()[1,:]

        self.ind_extremes   = {'t': [self.t0, self.tEnd], 'x': [self.xMin, self.xMax]}


        Data                = pd.read_csv(PathToExtremes+'/u0_par.csv')
        self.uMin           = Data.to_numpy()[0,:]
        self.uMax           = Data.to_numpy()[1,:]

        self.other_extremes = {'u0_par': [self.u0Min, self.u0Max]}

        self.from_extremes_to_ranges()

    #===========================================================================



    #===========================================================================
    def read_steps(self, ROMNet_fld):

        PathToExtremes = ROMNet_fld + '/database/Allen_Cahn/Steps/'

        Data                = pd.read_csv(PathToExtremes+'/t.csv')
        self.n_t            = Data.to_numpy()[0,0]
      
        Data                = pd.read_csv(PathToExtremes+'/x.csv')
        self.n_x            = Data.to_numpy()[0,0]
        
    #===========================================================================



    #===========================================================================
    def read_params(self, ROMNet_fld):

        PathToParams = ROMNet_fld + '/database/Allen_Cahn/Params/'

        nu = pd.read_csv(PathToParams+'/nu.csv').to_numpy()[0,0]
        xi = pd.read_csv(PathToParams+'/xi.csv').to_numpy()[0,0]
        
        self.params = [nu, xi]

    #===========================================================================