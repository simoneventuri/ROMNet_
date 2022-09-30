import numpy                    as np
from scipy.integrate import ode as scipy_ode
import pandas                   as pd
import tensorflow               as tf

from .system import System



class Generic(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):
        ROMNet_fld        = InputData.ROMNet_fld

        # Integrator time step
        # self.dt0          = 1.e-3
        # self.dt_str       = 1.
        # self.dt_max       = 0.02

        self.norm_output_flg       = InputData.norm_output_flg
        try:
            self.data_preproc_type = InputData.data_preproc_type
        except:
            self.data_preproc_type = None
        # Integration method
        # self.method       = 'bdf'


        # self.order        = [1]

        # self.ind_names    = ['x','v']
        # self.other_names  = ['t']

        # self.ind_labels   = ['t [s]']
        # self.other_labels = ['x [m]','v [m/s]']

        # self.get_variable_locations()

        self.fROM_anti   = None
        
    #===========================================================================



    #===========================================================================
    def f(self, t, y, arg):
        return np.matmul(arg, y)

    #===========================================================================



    #===========================================================================
    def jac(self, t, y, arg):
        return arg

    #===========================================================================



    #===========================================================================
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''

        r = scipy_ode(self.f, self.jac).set_integrator( 'vode', 
                           method=self.method, with_jacobian=True, atol=1.e-10 )
        r.set_initial_value(y_0, self.t0)
        r.set_f_params((self.K,))
        r.set_jac_params((self.K,))

        # Appending Data
        t = np.array([self.t0])
        y = np.expand_dims(y_0,0)

        dt = self.dt0
        while r.successful() and r.t <= self.tEnd:
            r.integrate(r.t+dt)
            t = np.vstack((t, np.expand_dims(r.t,0)))
            y = np.vstack((y, np.expand_dims(r.y,0)))
            dt = min(dt*self.dt_str, self.dt_max)

        return [t, y]

    #===========================================================================



    #===========================================================================
    def get_residual(self, ICs, t, grads):

        y, dy_dt = grads


        if (self.norm_output_flg):

            if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
                y = y * self.output_std + self.output_mean

            elif (self.data_preproc_type == '0to1'):
                y = y * self.output_range + self.output_min

            elif (self.data_preproc_type == 'range'):
                y = y * self.output_range

            elif (self.data_preproc_type == '-1to1'):
                y = (y + 1.)/2. * self.output_range + self.output_min

            elif (self.data_preproc_type == 'pareto'):
                y = y * np.sqrt(self.output_std) + self.output_mean


        dy_ct_dt = tf.matmul(y, self.K, transpose_b=True)


        if (self.norm_output_flg):

            if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
                dy_ct_dt /= self.output_std

            elif (self.data_preproc_type == '0to1'):
                dy_ct_dt /= self.output_range

            elif (self.data_preproc_type == 'range'):
                dy_ct_dt /= self.output_range

            elif (self.data_preproc_type == '-1to1'):
                dy_ct_dt /= self.output_range*2.

            elif (self.data_preproc_type == 'pareto'):
                dy_ct_dt /= np.sqrt(self.output_std)


        return dy_dt - dy_ct_dt

    #===========================================================================



    #===========================================================================
    def get_matrix(self):

        m, k, c = self.params
        return np.array([[0, 1.],[-k/m, -c/m]], dtype=np.float64)

    #===========================================================================



    #===========================================================================
    def read_extremes(self, ROMNet_fld):

        PathToExtremes = ROMNet_fld + '/database/MassSpringDamper/Extremes/'

        Data                = pd.read_csv(PathToExtremes+'/t.csv')
        self.t0             = Data.to_numpy()[0,:]
        self.tEnd           = Data.to_numpy()[1,:]

        self.ind_extremes   = {'t': [self.t0, self.tEnd]}


        Data                = pd.read_csv(PathToExtremes+'/xv.csv')
        self.xvMin          = Data.to_numpy()[0,:]
        self.xvMax          = Data.to_numpy()[1,:]

        self.other_extremes = {'x': [self.xvMin[0], self.xvMax[0]], 'v': [self.xvMin[1], self.xvMax[1]]}

        self.from_extremes_to_ranges()

    #===========================================================================



    #===========================================================================
    def read_params(self, ROMNet_fld):

        PathToParams = ROMNet_fld + '/database/MassSpringDamper/Params/'

        m = pd.read_csv(PathToParams+'/m.csv').to_numpy()[0,0]
        k = pd.read_csv(PathToParams+'/k.csv').to_numpy()[0,0]
        c = pd.read_csv(PathToParams+'/c.csv').to_numpy()[0,0]

        self.params = [m, k, c]

    #===========================================================================