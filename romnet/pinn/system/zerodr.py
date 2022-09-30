import numpy                          as np
from scipy.integrate import solve_ivp as scipy_ode
import pandas                         as pd
import tensorflow                     as tf
import cantera                        as ct

from .system import System



class ZeroDR(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):  

        path_to_data_fld       = InputData.path_to_data_fld
        ROMNet_fld             = InputData.ROMNet_fld
        try:
            self.ROM_type      = InputData.ROM_type
        except:
            self.ROM_type      = None
        self.NRODs             = InputData.NRODs

        self.mixture_file      = 'gri30.yaml'
        # self.fuel              = 'CH4:1.0'
        # self.oxidizer          = 'O2:1.0, N2:0.0'
        self.fuel              = 'H2:1.0'
        self.oxidizer          = 'O2:1.0, N2:4.0'

        self.T0                = 300.
        self.P0                = ct.one_atm
        self.eq_ratio0         = 1.


        self.norm_output_flg   = InputData.norm_output_flg
        self.data_preproc_type = InputData.data_preproc_type

        self.pinn_flg          = 'pts' in InputData.n_train
 
        self.order             = [1]
 
        self.ind_names         = ['t']
        self.other_names       = InputData.input_vars_all.copy()
        try:
            self.other_names.remove('t')
        except:
            pass
        self.ind_labels        = ['t [s]']
        self.other_labels      = ['PC_{0_{'+str(i)+'}}' for i in range(self.NRODs)]

        self.get_variable_locations()

        # Initial/final time
        self.read_extremes(ROMNet_fld)

        # Initialize Reactor
        self.initialize_reactor()

        self.read_params_ROM(path_to_data_fld)
        
        if (self.ROM_type == 'PCA'):
            self.f_call = self.f_pc
            self.n_dims = self.n_pc
        elif (self.ROM_type == 'Autoencoder'):
            self.f_call = self.f_pc
            self.n_dims = self.NRODs
        elif (self.ROM_type is None):
            self.f_call = self.f
            self.n_dims = len(self.other_names)

        self.n_batch    = InputData.batch_size 

    #===========================================================================



    #===========================================================================
    def f_orig(self, t, y_orig, ICs):
        n_points                   = y_orig.shape[0]

        dy_orig_dt                 = np.zeros_like(y_orig)
        for i_point in range(n_points):

            T                      = y_orig[i_point,0]
            Y                      = y_orig[i_point,1:]
            self.gas.TPY           = T, self.P0, Y
                          
            wdot                   = self.gas.net_production_rates

            dy_orig_dt[i_point,0]  = - np.dot(wdot, self.gas.partial_molar_enthalpies) / self.gas.cp / self.gas.density
            dy_orig_dt[i_point,1:] = wdot * self.gas.molecular_weights / self.gas.density
   
        return dy_orig_dt

    #===========================================================================



    # #===========================================================================
    # def f(self, t, y_masked, ICs):
    #     n_points                   = y_masked.shape[0]

    #     y_orig                     = np.zeros((n_points,self.n_species+1))
    #     y_orig[:,self.to_orig]     = y_masked
    #     dy_orig_dt                 = np.zeros_like(y_orig)
    #     for i_point in range(n_points):
    #         T                      = y_orig[i_point,0]
    #         Y                      = y_orig[i_point,1:]
    #         self.gas.TPY           = T, self.P0, Y
                          
    #         wdot                   = self.gas.net_production_rates

    #         dy_orig_dt[i_point,0]  = - np.dot(wdot, self.gas.partial_molar_enthalpies) / self.gas.cp / self.gas.density
    #         dy_orig_dt[i_point,1:] = wdot * self.gas.molecular_weights / self.gas.density
   
    #     dy_masked_dt               = dy_orig_dt[:,self.to_orig]

    #     return dy_masked_dt

    # #===========================================================================


    #===========================================================================
    def f(self, t, y_masked, ICs):
        n_points                     = y_masked.shape[0]

        y_orig                       = np.zeros((n_points,self.n_species+1))
        y_orig[:,self.to_orig]       = y_masked
        dy_masked_dt                 = np.zeros_like(y_masked)
        for i_point in range(n_points):
            T                        = y_orig[i_point,0]
            Y                        = y_orig[i_point,1:]
            self.gas.TPY             = T, self.P0, Y
                          
            wdot                     = self.gas.net_production_rates

            dy_masked_dt[i_point,0]  = - np.dot(wdot, self.gas.partial_molar_enthalpies) / self.gas.cp / self.gas.density
            dy_masked_dt[i_point,1:] = wdot[self.to_orig[1::]] * self.gas.molecular_weights[self.to_orig[1::]] / self.gas.density
   
        return dy_masked_dt

    #===========================================================================



    #===========================================================================
    def f_pc(self, t, y_pc, ICs):
        n_points                   = y_pc.shape[0]
        
        y_masked                   = np.matmul(y_pc, self.A) 
    
        y_orig                     = np.zeros((n_points,self.n_species+1))
        y_orig[:,self.to_orig]     = y_masked * self.D[0,:] + self.C[0,:]
        dy_orig_dt                 = np.zeros_like(y_orig)
        for i_point in range(n_points):
            T                      = y_orig[i_point,0]
            Y                      = np.maximum(y_orig[i_point,1:], 0.)
            self.gas.TPY           = T, self.P0, Y
                          
            wdot                   = self.gas.net_production_rates

            dy_orig_dt[i_point,0]  = - np.dot(wdot, self.gas.partial_molar_enthalpies) / self.gas.cp / self.gas.density
            dy_orig_dt[i_point,1:] = wdot * self.gas.molecular_weights / self.gas.density
   
        dy_masked_dt               = dy_orig_dt[:,self.to_orig] / self.D[0,:]
        dy_pc_dt                   = np.matmul(dy_masked_dt, self.AT)

        return dy_pc_dt

    #===========================================================================



    #===========================================================================
    def jac(self, t, y, arg):
        return arg

    #===========================================================================



    #===========================================================================
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''
        return None

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


        dy_ct_dt = self.f_call(t, y.numpy(), ICs.numpy()) #* np.exp(t[0].numpy())


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
    def read_extremes(self, ROMNet_fld):

        # PathToExtremes      = ROMNet_fld + '/database/MassSpringDamper/Extremes/'

        # Data                = pd.read_csv(PathToExtremes+'/t.csv')
        # self.t0             = Data.to_numpy()[0,:]
        # self.tEnd           = Data.to_numpy()[1,:]

        self.ind_extremes   = None #{'t': [self.t0, self.tEnd]}


        # Data                = pd.read_csv(PathToExtremes+'/xv.csv')
        # self.xvMin          = Data.to_numpy()[0,:]
        # self.xvMax          = Data.to_numpy()[1,:]

        self.other_extremes = None #{'x': [self.xvMin[0], self.xvMax[0]], 'v': [self.xvMin[1], self.xvMax[1]]}

        # self.from_extremes_to_ranges()

        self.other_ranges = None

    #===========================================================================



    #===========================================================================
    def read_params_ROM(self, path_to_data_fld):

        if (self.ROM_type == 'PCA'):
            PathToParams    = path_to_data_fld + '/ROM/'
        # # elif (self.ROM_type ):
        # #     PathToParams    = path_to_data_fld + '/../'+str(self.NRODs)+'PC/ROM/'
        # #     self.OutputVars = list(pd.read_csv(path_to_data_fld+'/train/ext/CleanVars.csv', header=None).to_numpy()[0,:])

        # if (self.NRODs > 0):
            try:
                self.A         = pd.read_csv(PathToParams+'/A.csv', header=None).to_numpy()
                self.C         = pd.read_csv(PathToParams+'/C.csv', header=None).to_numpy().T
                self.D         = pd.read_csv(PathToParams+'/D.csv', header=None).to_numpy().T
                self.AT        = self.A.T
                self.n_pc      = self.A.shape[0]

                self.to_orig   = pd.read_csv(PathToParams+'/ToOrig_Mask.csv',     header=None).to_numpy(int)[:,0]
                self.n_mask    = len(self.to_orig)
            except:
                self.n_pc      = self.NRODs
                pass

        elif (self.ROM_type is None):
            self.to_orig   = pd.read_csv(path_to_data_fld+'/ToOrig_Mask.csv', header=None).to_numpy(int)[:,0]
            self.n_mask    = len(self.to_orig)

    #===========================================================================



    #===========================================================================
    def initialize_reactor(self):

        ### Create Mixture
        gas            = ct.Solution(self.mixture_file)
        self.n_species = gas.n_species

        ### Create Reactor
        #gas.TP         = self.T0, self.P0
        #gas.set_equivalence_ratio(self.eq_ratio0, self.fuel, self.oxidizer)
        self.gas       = gas


    #===========================================================================
    


    #===========================================================================
    def fROM_anti(self):

        def fROM_anti_PCA(y_pc):
            y_masked = tf.matmul(y_pc, self.A) #* self.D + self.C
            return y_masked

        if (self.ROM_type):
            return None
        else:
            return None #fROM_anti_PCA 
    #===========================================================================



    #===========================================================================
    def preprocess_data(self, all_data, xstat):

        for i, now_data in enumerate(all_data):
            for data_id, xyi_data in now_data.items():

                # all_data[i][data_id][0]['HH'] = (all_data[i][data_id][0]['HH'] - xstat['min'].to_numpy()[0]) / (xstat['max'].to_numpy()[0] - xstat['min'].to_numpy()[0])
                # all_data[i][data_id][1]['HH'] = (all_data[i][data_id][1]['HH'] - xstat['min'].to_numpy()[0]) / (xstat['max'].to_numpy()[0] - xstat['min'].to_numpy()[0])

                # for var in list(all_data[i][data_id][0].columns)[1:]:
                #     all_data[i][data_id][0][var] = np.log10(all_data[i][data_id][0][var])          
                #     all_data[i][data_id][1][var] = np.log10(all_data[i][data_id][1][var])   
                
                all_data[i][data_id][0] = (all_data[i][data_id][0] - xstat['mean'].to_numpy()) / np.sqrt(xstat['std'].to_numpy())
                all_data[i][data_id][1] = (all_data[i][data_id][1] - xstat['mean'].to_numpy()) / np.sqrt(xstat['std'].to_numpy())

        return all_data       

    #===========================================================================



# #=======================================================================================================================================
# class AutoEncoderLayer(tf.keras.layers.Layer):

#     def __init__(self, path_to_data_fld, NVars, trainable_flg=False, name='AutoEncoderLayer'):
#         super(AutoEncoderLayer, self).__init__(name=name, trainable=False)

#         self.path_to_data_fld = path_to_data_fld
#         Data               = pd.read_csv(self.path_to_data_fld+'/train/ext/Output_MinMax.csv')
#         var_min            = Data.to_numpy()[:,0]
#         var_max            = Data.to_numpy()[:,1]

#         self.HH_min        = var_min[0]
#         self.HH_max        = var_max[0]
#         self.HH_range      = self.HH_max - self.HH_min
#         self.NVars         = NVars
#         self.trainable_flg = trainable_flg

#     def call(self, inputs):

#         inputs_unpack    = tf.split(inputs, [1,self.NVars-1], axis=1)

#         inputs_unpack[0] = (inputs_unpack[0] - self.HH_min) / self.HH_range

#         #inputs_unpack[1] = tf.experimental.numpy.log10(inputs_unpack[1])
#         inputs_unpack[1] = tf.math.log(inputs_unpack[1])
        
#         return tf.concat(inputs_unpack, axis=1)

# #=======================================================================================================================================



# #=======================================================================================================================================
# class AntiAutoEncoderLayer(tf.keras.layers.Layer):

#     def __init__(self, path_to_data_fld, NVars, trainable_flg=False, name='AntiAutoEncoderLayer'):
#         super(AntiAutoEncoderLayer, self).__init__(name=name, trainable=False)

#         self.path_to_data_fld = path_to_data_fld
#         Data               = pd.read_csv(self.path_to_data_fld+'/train/ext/Output_MinMax.csv')
#         var_min            = Data.to_numpy()[:,0]
#         var_max            = Data.to_numpy()[:,1]

#         self.HH_min        = var_min[0]
#         self.HH_max        = var_max[0]
#         self.HH_range      = self.HH_max - self.HH_min
#         self.NVars         = NVars
#         self.trainable_flg = trainable_flg

#     def call(self, inputs):

#         inputs_unpack    = tf.split(inputs, [1,self.NVars-1], axis=1)

#         inputs_unpack[0] = inputs_unpack[0] * self.HH_range + self.HH_min

#         #inputs_unpack[1] = 10**(inputs_unpack[1])
#         inputs_unpack[1] = tf.math.exp(inputs_unpack[1])
        
#         return tf.concat(inputs_unpack, axis=1)

# #=======================================================================================================================================