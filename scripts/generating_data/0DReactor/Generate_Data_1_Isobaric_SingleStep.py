import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time

import pyDOE
import cantera as ct
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
WORKSPACE_PATH = os.getcwd()+'/../../../../../'

# import matplotlib.pyplot as plt
# plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

from joblib import Parallel, delayed
import multiprocessing



##########################################################################################
### Input Data

### HYDROGEN
OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_2Cases_H2_SmallSteps/'
Fuel0              = 'H2:1.0'         
Oxydizer0          = 'O2:1.0, N2:4.0'
#Deltat             = 1.e-5
DeltatMax          = -4
DeltatMin          = -6
#tEnd               = 1.e-1
KeepVec            = None #['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3','NNH','NO','NO2','N2O','HNO','N2']

# ### METHANE
# OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4/'
# Fuel0              = 'CH4:1.0'
# Oxydizer0          = 'O2:0.21, N2:0.79'
# t0                 = 1.e-6
# tEnd               = 1.e2
# KeepVec            = None

MixtureFile        = 'gri30.yaml'
P0                 = ct.one_atm

NtInt              = 5
Integration        = 'Canteras'
delta_T_max        = 1.
# Integration        = ''
# rtol               = 1.e-12
# atol               = 1.e-8
# SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

# FIRST TIME
DirName            = 'train_file'
#n_ics              = 20
# T0Exts             = np.array([980., 1020], dtype=np.float64)
# EqRatio0Exts       = np.array([0.98, 1.02], dtype=np.float64)
T0Exts             = 'File' #np.array([1000., 2000.], dtype=np.float64)
#ExtsFile           = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_10Cases_H2/Orig/MinMax.csv'
ExtsFile           = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_1000Cases_H2//Orig/train/ext/Uniformized.csv'
EqRatio0Exts       = None #np.array([.5, 4.], dtype=np.float64)
X0Exts             = None #np.array([0.05, 0.95], dtype=np.float64)
SpeciesVec         = None #['H2','H','O','O2','OH','N','NH','NO','N2']

# ## SECOND TIME
# DirName            = 'test'
# n_ics              = 10
# # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
# T0Exts             = np.array([990., 1010.], dtype=np.float64)
# EqRatio0Exts       = np.array([0.99, 1.01], dtype=np.float64)
# X0Exts             = None
# SpeciesVec         = None
# NPerT0             = 10000

n_processors         = 1



##########################################################################################

try:
    os.makedirs(OutputDir)
except:
    pass
# try:
#     os.makedirs(FigDir)
# except:
#     pass
try:
    os.makedirs(OutputDir+'/Orig/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+DirName+'/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+DirName+'/ext/')
except:
    pass




##########################################################################################
### Defining ODE and its Parameters
def IdealGasConstPressureReactor_SciPY(t, y):
    #print(t)

    Y          = y[1:]
    # YEnd     = np.array([1.-np.sum(y[1:])], dtype=np.float64)
    # Y        = np.concatenate((y[1:], YEnd), axis=0)
    gas_.TPY = y[0], P_, Y
    
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    ydot[1:] = wdot * gas_.molecular_weights / gas_.density
    # ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / gas_.density
    
    return ydot


def IdealGasConstPressureReactor(t, T, Y):

    gas_.TP   = T, P_
    if (SpeciesVec):
        gas_sub = gas_[SpeciesVec]
    else:
        gas_sub = gas_
    gas_sub.Y = Y 

    
    wdot     = gas_sub.net_production_rates

    Tdot     = - np.dot(wdot, gas_sub.partial_molar_enthalpies) / gas_sub.cp / gas_sub.density
    Ydot     = wdot * gas_sub.molecular_weights / gas_sub.density

    HR       = - np.dot(gas_sub.net_production_rates,gas_sub.partial_molar_enthalpies)

    return Tdot*t, Ydot*t, HR


def IdealGasReactor_SciPY(t, y):
    #print(t)

    yPos     = np.maximum(y, 0.)
    ySum     = np.minimum(np.sum(y[1:]), 1.)
    YEnd     = np.array([1.-ySum], dtype=np.float64)
    Y        = np.concatenate((y[1:], YEnd), axis=0)
    gas_.TDY = y[0], density_, Y
    
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_int_energies) / gas_.cv / density_
    ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / density_
    
    return ydot


def IdealGasReactor(t, T, Y):

    gas_.TDY = T, density_, np.maximum(Y, 0.)
    
    wdot     = gas_.net_production_rates

    Tdot     = - np.dot(wdot, gas_.partial_molar_int_energies) / gas_.cv / density_
    Ydot     = wdot * gas_.molecular_weights / density_

    HR       = - np.dot(gas_.net_production_rates, gas_.partial_molar_enthalpies)
    
    return Tdot, Ydot, HR




def integration_(iIC, n_processors, OutputDir, DirName, ICs, MixtureFile, SpeciesVec, Mask_):

    #try:
    
    P0       = ICs[iIC,0]
    T0       = ICs[iIC,1]
    Y0       = ICs[iIC,2::]

    ### Create Reactor
    gas     = ct.Solution(MixtureFile)
    gas.TP  = T0, P0


    SpecDict = {}
    for iS, Spec in enumerate(SpeciesVec):
        SpecDict[Spec] = ICs[iIC,iS+2]
    gas.Y    = SpecDict


    r       = ct.IdealGasConstPressureReactor(gas)
    sim     = ct.ReactorNet([r])
    sim.verbose = False

    gas_    = gas
    mass_   = r.mass
    # print('   Mass = ', mass_)
    density_= r.density
    P_      = P0
    y0      = np.array(np.hstack((gas_.T, gas_.Y)), dtype=np.float64)



    ############################################################################
    # E
    tVec = np.append([0.0], np.logspace(-8, -4, NtInt))
    #############################################################################


    gas_             = gas
    if (SpeciesVec):
        gas_kept     = gas[SpeciesVec]
    else:
        gas_kept     = gas
    states           = ct.SolutionArray(gas_kept, 1, extra={'t': [0.0]})

    
    #r.set_advance_limit('temperature', delta_T_max)
    TT               = r.T
    YY               = r.thermo.Y[Mask_]
    Vec              = np.concatenate(([TT],YY), axis=0)
    Mat              = np.array(Vec[np.newaxis,...])
    tVecFinal        = np.array(tVec, dtype=np.float64)

   
    ### Integrate
    it               = 0
    for t in tVecFinal[it:]:

        sim.advance(t)
        TT                   = r.T
        YY                   = r.thermo.Y[Mask_]
        Vec                  = np.concatenate(([TT],YY), axis=0)
        if (it == 0):
            Mat              = np.array(Vec[np.newaxis,...])
        else:
            Mat              = np.concatenate((Mat, Vec[np.newaxis,...]), axis=0)

        it+=1 


    if (iIC<n_processors):
        WrtFlg = "w"
    else:
        WrtFlg = "ab"

    y00      = np.concatenate([tVecFinal[1::][...,np.newaxis],  np.repeat(np.clip(Mat[0,:][np.newaxis,...], 1.e-30, 1e10), NtInt, axis=0)], axis=1)
    #y00      = np.concatenate(([tVecFinal[0]],  np.log10(np.clip(Mat[0, :], 1.e-30, 1e10))), axis=0)[np.newaxis,...]
    FileName = OutputDir+'/Orig/'+DirName+'/ext/y0.csv.'+str((iIC%n_processors)+1)
    with open(FileName, WrtFlg) as f:
        if (iIC<n_processors):
            Header0  = 't,T'
            for Keep in SpeciesVec:
                Header0 += '0,'+Keep   
            f.write(Header0+"0\n")
        np.savetxt(f, y00, delimiter=',')

    yEnd     = np.concatenate([tVecFinal[1::][...,np.newaxis], np.clip(Mat[1::,:], 1.e-30, 1e10)], axis=1)
    #yEnd     = np.concatenate(([tVecFinal[-1]], np.log10(np.clip(Mat[-1,:], 1.e-30, 1e10))), axis=0)[np.newaxis,...]
    FileName = OutputDir+'/Orig/'+DirName+'/ext/yEnd.csv.'+str((iIC%n_processors)+1)
    with open(FileName, WrtFlg) as f:
        if (iIC<n_processors):
            Header   = 't,T'
            for Keep in SpeciesVec:
                Header  += ','+Keep
            f.write(Header+"\n")
        np.savetxt(f, yEnd, delimiter=',')

    # except:
    #     pass

        

##########################################################################################
### Generating Training Data


if (DirName == 'train'):

    DataDF     = pd.read_csv(ExtsFile, header=0)
    SpeciesVec = list(DataDF.columns)[1::]
    MinVals    = DataDF.to_numpy()[0,:]
    MaxVals    = DataDF.to_numpy()[1,:]
    MinVals    = np.log10(MinVals)
    MaxVals    = np.log10(MaxVals)

    NDims      = len(SpeciesVec)+1
    ICs        = pyDOE.lhs(NDims, samples=n_ics, criterion='center')
    for i in range(NDims):
        ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]
        ICs[:,i] = 10.**(ICs[:,i])
    ICs = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

    ### Writing Initial Temperatures
    FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
    Header   = 'P,T,'+','.join(SpeciesVec)
    np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')


elif (DirName == 'train_file'):

    DataDF     = pd.read_csv(ExtsFile, header=0)
    n_ics      = len(DataDF)
    SpeciesVec = list(DataDF.columns)[1::]
    ICs        = DataDF.to_numpy()
    NDims      = len(SpeciesVec)+1
    ICs        = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

    ### Writing Initial Temperatures
    FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
    Header   = 'P,T,'+','.join(SpeciesVec)
    np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')


elif (DirName == 'test'):
    # NDims    = 2
    # ICs      = np.zeros((n_ics,NDims))
    # # ICs[:,0] = [2.5, 1.9, 3.5, 1., 3.6]
    # # ICs[:,1] = [1200., 1900., 1300., 1600., 1700.]
    # ICs[:,0] = [0.0.8, 0.9, 1.0, 1.1, 1.2]
    # ICs[:,1] = [1300., 1200., 1400., 1500., 1250.]
    # ICs = np.concatenate([P0*np.ones((n_ics,1)), ICs], axis=1)
    MinVals = np.array([EqRatio0Exts[0], T0Exts[0]], dtype=np.float64)
    MaxVals = np.array([EqRatio0Exts[1], T0Exts[1]], dtype=np.float64)
    NDims   = 2

    ICs     = pyDOE.lhs(2, samples=n_ics, criterion='center')

    for i in range(NDims):
        ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]
    ICs = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

    ### Writing Initial Temperatures
    FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
    Header   = 'P,EqRatio,T'
    np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')



P_ = P0


### Create Mixture
gas     = ct.Solution(MixtureFile)

Mask_ = []
if (SpeciesVec):
    for Keep in SpeciesVec:
        Mask_.append(gas.species_names.index(Keep))
    Mask_ = np.array(Mask_)
else:
    Mask_ = np.arange(len(gas.species_names))


results = Parallel(n_jobs=n_processors)(delayed(integration_)(iIC, n_processors, OutputDir, DirName, ICs, MixtureFile, SpeciesVec, Mask_) for iIC in range(n_ics))





### Writing Results
NSpec        = gas.n_species
Header       = 't,T'
SpeciesNames = []
for iSpec in range(NSpec):
    Header += ','+gas.species_name(iSpec)
    SpeciesNames.append(gas.species_name(iSpec))

# FileName = OutputDir+'/orig_data/States.csv.'+str(iIC+1)
# np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')
print('Original (', len(SpeciesNames), ') Species: ', SpeciesNames)

VarsName    = ['T']+SpeciesNames
if (DirName == 'train'):
    
    ToOrig       = []
    OrigVarNames = ['T']+SpeciesNames
    for Var in VarsName:
        ToOrig.append(OrigVarNames.index(Var))
    ToOrig = np.array(ToOrig, dtype=int)

    FileName = OutputDir+'/Orig/ToOrig_Mask.csv'
    np.savetxt(FileName, ToOrig, delimiter=',')


    FileName = OutputDir+'/Orig/'+DirName+'/ext/CleanVars.csv'
    StrSep = ','
    with open(FileName, 'w') as the_file:
        the_file.write(StrSep.join(VarsName)+'\n')

# ##########################################################################################