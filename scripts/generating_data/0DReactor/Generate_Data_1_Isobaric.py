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
import itertools


# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ### Input Data for Test Case 3 in S. Venturi & T. Casey, "SVD Perspectives for Augmenting DeepONet Flexibility and Interpretability", 2022

# ### HYDROGEN
# OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2/'
# Fuel0              = 'H2:1.0'         
# Oxydizer0          = 'O2:1.0, N2:4.0'
# t0                 = 1.e-7
# # Deltat             = 1.e-5
# DeltatMax          = None #7
# DeltatMin          = None #4
# tEnd               = 1.e0
# SpeciesVec         = ['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3','NNH','NO','NO2','N2O','HNO','N2']

# # ### METHANE
# # OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4/'
# # Fuel0              = 'CH4:1.0'
# # Oxydizer0          = 'O2:0.21, N2:0.79'
# # t0                 = 1.e-6
# # tEnd               = 1.e2
# # KeepVec            = None

# MixtureFile        = 'gri30.yaml'
# #MixtureFile        = WORKSPACE_PATH + '/Modify_CANTERA/Run_1/gri30'
# P0                 = ct.one_atm

# NtInt              = 500
# Integration        = 'Canteras'
# delta_T_max        = 5.e0
# # Integration        = ''
# # rtol               = 1.e-12
# # atol               = 1.e-8
# # SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

# # FIRST TIME
# DirName            = 'train'
# n_ics              = 500
# n_samples          = 1
# T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# EqRatio0Exts       = np.array([0.5, 4.0], dtype=np.float64)
# # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
# NPerT0             = 500
# NoiseStd           = None #5.e-2

# # ## SECOND TIME
# # DirName            = 'test'
# # n_ics              = 10
# # n_samples          = 1
# # # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # # EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
# # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # EqRatio0Exts       = np.array([0.50, 4.0], dtype=np.float64)
# # X0Exts             = None
# # SpeciesVec         = None
# # NPerT0             = 1000
# # NoiseStd           = None

# n_processors       = 8

# ##########################################################################################
# ##########################################################################################
# ##########################################################################################



##########################################################################################
### Input Data

### HYDROGEN
OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2/'
Fuel0              = 'H2:1.0'         
Oxydizer0          = 'O2:1.0, N2:4.0'
t0                 = 1.e-7
# Deltat             = 1.e-5
DeltatMax          = None #7
DeltatMin          = None #4
tEnd               = 1.e0
SpeciesVec         = ['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3','NNH','NO','NO2','N2O','HNO','N2']

# ### METHANE
# OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4/'
# Fuel0              = 'CH4:1.0'
# Oxydizer0          = 'O2:0.21, N2:0.79'
# t0                 = 1.e-6
# tEnd               = 1.e2
# KeepVec            = None

MixtureFile        = 'gri30.yaml'
#MixtureFile        = WORKSPACE_PATH + '/Modify_CANTERA/Run_1/gri30'
P0                 = ct.one_atm

NtInt              = 500
Integration        = 'Canteras'
delta_T_max        = 5.e0
# Integration        = ''
# rtol               = 1.e-12
# atol               = 1.e-8
# SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

# # FIRST TIME
# DirName            = 'train'
# n_ics              = 500
# n_samples          = 1
# T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# EqRatio0Exts       = np.array([0.5, 4.0], dtype=np.float64)
# # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
# NPerT0             = 500
# NoiseStd           = None #5.e-2

## SECOND TIME
DirName            = 'test'
n_ics              = 10
n_samples          = 1
# T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
T0Exts             = np.array([1000., 2000.], dtype=np.float64)
EqRatio0Exts       = np.array([0.5, 4.0], dtype=np.float64)
X0Exts             = None
SpeciesVec         = ['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3','NNH','NO','NO2','N2O','HNO','N2']
NPerT0             = 500
NoiseStd           = None


n_processors       = 8

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
    gas_sub = gas_
    gas_sub.Y = Y 

    
    wdot     = gas_sub.net_production_rates

    Tdot     = - np.dot(wdot, gas_sub.partial_molar_enthalpies) / gas_sub.cp / gas_sub.density
    Ydot     = wdot * gas_sub.molecular_weights / gas_sub.density

    # HR       = - np.dot(gas_sub.net_production_rates,gas_sub.partial_molar_enthalpies)

    return Tdot*t, Ydot*t #, HR


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



def integration_(i_tot, zipp, n_samples, OutputDir, DirName, ICs, MixtureFile, Fuel0, Oxydizer0, NPerT0, t0, tEnd, DeltatMin, DeltatMax, delta_T_max, SpeciesVec, Mask_, iTot):
    iIC      = zipp[0]
    i_sample = zipp[1]

    P0       = ICs[iIC,0]
    EqRatio0 = ICs[iIC,1]
    T0       = ICs[iIC,2]
    print('Pressure = ', P0, 'Pa; EqRatio0 = ', EqRatio0, '; Temperature = ', T0, 'K')


    ### Create Mixture
    if (n_samples > 1):
        MixtureFile_ = MixtureFile + '_' + str(i_sample+1) + '.yaml'
    else:
        MixtureFile_ = MixtureFile
    gas = ct.Solution(MixtureFile_)

    ### Create Reactor
    gas.TP  = T0, P0

    gas.set_equivalence_ratio(EqRatio0, Fuel0, Oxydizer0)

    r       = ct.IdealGasConstPressureReactor(gas)
    sim     = ct.ReactorNet([r])
    sim.verbose = False

    gas_    = gas
    mass_   = r.mass
    # print('   Mass = ', mass_)
    density_= r.density
    P_      = P0
    y0      = np.array(np.hstack((gas_.T, gas_.Y)), dtype=np.float64)
    # y0      = np.array(np.hstack((gas_.T, gas_.Y[0:-1])), dtype=np.float64)

    ############################################################################
    # ### Initialize Integration 
    # tAuto    = 10**( (8.75058755*(1000/T0) -9.16120796) )
    # tMin     = tAuto * 1.e-1
    # tMax     = tAuto * 1.e1
    # dt0      = tAuto * 1.e-3

    # tStratch = 1.01
    # tVec     = [0.0]
    # t        = tMin
    # dt       = dt0
    # while (t <= tMax):
    #     tVec.append(t)
    #     t  =   t + dt
    #     dt = dt0 * tStratch
    ############################################################################

    ############################################################################
    # A
    #tVec     = np.append([0., 1.e-14], np.logspace(np.log10(t0), np.log10(tEnd), NtInt-2))
    tVec     = np.append([1.e-14], np.logspace(np.log10(t0), np.log10(tEnd), NtInt-1))
    #tVec     = np.logspace(np.log10(t0), np.log10(tEnd), NtInt)
    
    # B
    #tVec     = np.linspace(t0, tEnd, NtInt)
    
    # C
    # NtInt    = int(tEnd/Deltat)
    # t0       = np.random.rand() * Deltat
    # tVec     = np.append([0.], np.arange(NtInt-1) * Deltat + t0)

    # # D
    # tVec = []
    # t    = 0.
    # while (t<=tEnd):
    #     tVec.append(t)
    #     #t += np.random.rand() * (DeltatMax-DeltatMin)+DeltatMin
    #     t += 10.**( - ( DeltatMax - np.random.rand() * (DeltatMax-DeltatMin) ) )
    #############################################################################


    gas_             = gas
    gas_kept         = gas
    states           = ct.SolutionArray(gas_kept, 1, extra={'t': [0.0]})

    
    if (Integration == 'Canteras'):
        #r.set_advance_limit('temperature', delta_T_max)
        TT               = r.T
        YY               = r.thermo.Y[Mask_]
        Vec              = np.concatenate(([TT],YY), axis=0)
        # TTdot, YYdot, HR = IdealGasConstPressureReactor(tVec[0], TT, YY)
        # Vecdot           = np.concatenate(([TTdot],YYdot), axis=0)
        Mat              = np.array(Vec[np.newaxis,...])
        # Source           = np.array(Vecdot[np.newaxis,...])
        it0              = 1
        tVecFinal        = np.array(tVec, dtype=np.float64)
        # HRVec            = [HR]
    else:
        output           = solve_ivp( IdealGasConstPressureReactor_SciPY, (tVec[0],tVec[-1]), y0, method=SOLVER, t_eval=tVec, rtol=rtol, atol=atol )
        it0              = 0
        tVecFinal        = output.t
        # HRVec            = []

    ### Integrate
    it           = it0
    for t in tVecFinal[it:]:

        if (Integration == 'Canteras'):
            sim.advance(t)
            TT               = r.T
            YY               = r.thermo.Y[Mask_]
        else:
            TT               = output.y[0,it]
            YY               = output.y[1:,it]
            # YY               = np.concatenate((output.y[1:,it], [1.0-np.sum(output.y[1:,it])]), axis=0)

        Vec                  = np.concatenate(([TT],YY), axis=0)

        # TTdot, YYdot, HR     = IdealGasConstPressureReactor(t, TT, YY)
        # Vecdot               = np.concatenate(([TTdot],YYdot), axis=0)

        if (it == 0):
            Mat              = np.array(Vec[np.newaxis,...])
            # Source           = np.array(Vecdot[np.newaxis,...])
        else:
            Mat              = np.concatenate((Mat, Vec[np.newaxis,...]),       axis=0)
            # Source           = np.concatenate((Source, Vecdot[np.newaxis,...]), axis=0)

        # HRVec.append(HR)
        it+=1 
        


    ### Storing Results
    Nt  = len(tVecFinal)
    if (Nt < NPerT0):
        Mask = np.arange(Nt)
        Ntt  = Nt
    else:
        Mask = np.linspace(0,Nt-1,NPerT0, dtype=int)
        Ntt  = NPerT0


    if (NoiseStd):
        Mat_ = np.maximum(Mat, 1.e-30)
        Mat_ = np.log10(Mat_)
        C    = Mat_.mean(axis=0)
        D    = Mat_.std(axis=0)
        Mat_ = ( Mat_ - C ) / D
        Mat_ = Mat_ * np.random.normal(1., NoiseStd, size=Mat_.shape)
        Mat_ = Mat_ * D + C
        Mat  = 10**Mat_


    yTemp        = np.concatenate((tVecFinal[Mask,np.newaxis], Mat[Mask,:]), axis=1)
    # ySourceTemp  = np.concatenate((tVecFinal[Mask,np.newaxis], Source[Mask,:]), axis=1)
        
    iTot[iIC]    = Ntt
        

    ### Writing Results
    NSpec        = gas.n_species
    Header       = 't,T'
    SpeciesNames = []
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)
        SpeciesNames.append(gas.species_name(iSpec))

    # FileName = OutputDir+'/orig_data/States.csv.'+str(iIC+1)
    # np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')


    ### Writing Results
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(i_tot+1)

    DF = pd.DataFrame(yTemp, columns=['t','T']+SpeciesVec)
    DF.to_csv(FileName, index=False)
    #np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/Orig/'+DirName+'/ext/ySource.csv.'+str(iIC+1)
    # np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iIC+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')



##########################################################################################
### Generating Training Data


if (DirName == 'train'):

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



### Create Mixture
if (n_samples > 1):
    MixtureFile_ = MixtureFile + '_1.yaml'
else:
    MixtureFile_ = MixtureFile
gas = ct.Solution(MixtureFile_)

Mask_ = []
if (SpeciesVec):
    for Keep in SpeciesVec:
        Mask_.append(gas.species_names.index(Keep))
    Mask_      = np.array(Mask_)
else:
    Mask_      = np.arange(len(gas.species_names))
    SpeciesVec = list(gas.species_names)
FileName = OutputDir+'/Orig/ToOrig_Mask.csv'
np.savetxt(FileName, Mask_, delimiter=',')



iStart          = np.zeros(n_ics)
iEnd            = np.zeros(n_ics)
iTot            = np.zeros(n_ics)

if (n_processors == 1):
    i_tot = 0
    for iIC in range(n_ics):
        for i_sample in range(n_samples):
            zipp = (iIC,i_sample)
            integration_(i_tot, zipp, n_samples, OutputDir, DirName, ICs, MixtureFile, Fuel0, Oxydizer0, NPerT0, t0, tEnd, DeltatMin, DeltatMax, delta_T_max, SpeciesVec, Mask_, iTot)
            i_tot += 1
else:
    zipps   = list(itertools.product(np.arange(n_ics),np.arange(n_samples)))
    print(zipps)
    results = Parallel(n_jobs=n_processors)(delayed(integration_)(i_tot, zipp, n_samples, OutputDir, DirName, ICs, MixtureFile, Fuel0, Oxydizer0, NPerT0, t0, tEnd, DeltatMin, DeltatMax, delta_T_max, SpeciesVec, Mask_, iTot) for i_tot, zipp in enumerate(zipps))

# FileName = OutputDir+'/Orig/'+DirName+'/ext/SimIdxs.csv'
# Header   = 'iStart,iEnd'
# np.savetxt(FileName, np.concatenate((iStart[...,np.newaxis], iEnd[...,np.newaxis]), axis=1), delimiter=',', header=Header, comments='')



VarsName = ['T'] + SpeciesVec
FileName = OutputDir+'/Orig/'+DirName+'/ext/CleanVars.csv'
StrSep   = ','
with open(FileName, 'w') as the_file:
    the_file.write(StrSep.join(VarsName)+'\n')

##########################################################################################