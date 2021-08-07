import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
WORKSPACE_PATH  = os.environ['WORKSPACE_PATH']
plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

import cantera as ct
from scipy.integrate import solve_ivp


##########################################################################################
### Input Data

OutputDir          = WORKSPACE_PATH + '/ROMNet/Data_10_0DReact_EASY/'
FigDir             = OutputDir + '/fig/'

MixtureFile        = 'gri30.yaml'
fuel               = "CH4"
oxidizer           = "O2:0.21,N2:0.77,H:0.02"

P0                 = ct.one_atm
EqRatio0           = 1.

NTs                = 10
T0Vec              = np.logspace(np.log10(2000), np.log10(3000), NTs) # [2.e-5]
NPerT0             = 5000

dt0Vec             = [1.e-4, 1.e-6]
tMaxVec            = [5.e-2, 5.e-4]
Nt                 = NPerT0*2
tStratch           = 1.
##########################################################################################

try:
    os.makedirs(OutputDir)
except:
    pass
try:
    os.makedirs(FigDir)
except:
    pass
try:
    os.makedirs(OutputDir+'/orig_data/')
except:
    pass




##########################################################################################
### Defining ODE and its Parameters
def ReactorOde(t, y, gas_, P0):

    Y        = y[1:]

    gas_.TPY = y[0], P0, Y
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    ydot[1:] = wdot * gas_.molecular_weights / gas_.density

    return ydot


def ReactorOde_CVODE(t, y):
    #print(t)

    #YEnd     = np.array([1.-np.sum(y[1:])], dtype=np.float64)
    #Y        = np.concatenate((y[1:], YEnd), axis=0)
    Y        = y[1:]

    gas_.TPY = y[0], P0, Y
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    #ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / gas_.density
    ydot[1:] = wdot * gas_.molecular_weights / gas_.density

    return ydot



##########################################################################################
### Generating Training Data

### Writing Residence Times
FileName = OutputDir+'/orig_data/T0s.csv'
np.savetxt(FileName, T0Vec)



### Iterating Over Residence Times
DataMat  = None
iStart   = np.zeros(len(T0Vec))
iEnd     = np.zeros(len(T0Vec))
iSim     = 0
for T0 in T0Vec:
    print('Temperature = ', T0)
    

    ### Create Mixture
    gas      = ct.Solution(MixtureFile)

    gas.TP  = T0, P0
    gas.set_equivalence_ratio(EqRatio0, fuel, oxidizer)
    r       = ct.IdealGasConstPressureReactor(gas)
    Y0      = gas.Y
    NSpec   = gas.n_species


    ### Create Reactor
    sim         = ct.ReactorNet([r])
    sim.verbose = False

    # limit advance when temperature difference is exceeded
    delta_T_max =  10.
    r.set_advance_limit('temperature', delta_T_max)
    states      = ct.SolutionArray(gas, extra=['t'])
    gas_        = gas
    mass        = r.mass


    ### Initialize Integration 
    aa         = (T0 - T0Vec[0]) / (T0Vec[-1] - T0Vec[0])
    dt0        = aa * (dt0Vec[1]  - dt0Vec[0])  + dt0Vec[0]
    tMax       = aa * (tMaxVec[1] - tMaxVec[0]) + tMaxVec[0]
    print('dt0  = ', dt0)
    print('tMax = ', tMax)

    tout       = [0.]
    tout       = np.concatenate((np.array(tout, dtype=np.float64), np.linspace(dt0, tMax, Nt-1, dtype=np.float64)), axis=0)
    #tout       = np.linspace(dt0, tMax, Nt-1)

    Vec        = np.concatenate(([r.T],r.thermo.Y), axis=0)
    Mat        = np.array(Vec[np.newaxis,...])
    Source     = ReactorOde(sim.time, Vec, gas_, P0)[np.newaxis,...]

    # print('{:10s} {:10s} {:10s} {:14s}'.format('t [s]', 'T [K]', 'P [Pa]', 'u [J/kg]'))
    # print('{:10.3e} {:10.3f} {:10.3f} {:14.6f}'.format(0.0, r.T, r.thermo.P, r.thermo.u))
    for tnow in tout[1:]:
        sim.advance(tnow)
        states.append(r.thermo.state, t=sim.time)
        # print('{:10.3e} {:10.3f} {:10.3f} {:14.6f}'.format(sim.time, r.T, r.thermo.P, r.thermo.u))
        #Vec  = np.concatenate(([r.T],r.thermo.Y), axis=0)
        #dydt = ReactorOde(sim.time, Vec, gas_, P0)


    ### Integrate
    NTott    = len(states.T)
    print(NTott)
    #JJTauMat = np.zeros((NTott, output.y.shape[0]))
    for it in range(NTott):
        t   = states.t[it]
        T   = states.T[it] 
        Y   = states.Y[it,:]
        Vec = np.concatenate(([T],Y), axis=0)

        Mat        = np.concatenate((Mat, Vec[np.newaxis,...]), axis=0)
        SourceVec  = ReactorOde(sim.time, Vec, gas_, P0)[np.newaxis,...]
        Source     = np.concatenate((Source, SourceVec), axis=0)




    ### Storing Results
    Nt  = len(states.t)
    if (Nt < NPerT0):
        Mask = np.arange(Nt)
        Ntt  = Nt
    else:
        Mask = np.linspace(0,Nt-1,NPerT0, dtype=int)
        Ntt  = NPerT0

    if (iSim == 0):
        T0All        = np.ones(Ntt)*T0
        yTemp        = np.concatenate((tout[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = yTemp

        ySourceTemp  = np.concatenate((tout[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = ySourceTemp
        
        iStart[iSim] = 0
        iEnd[iSim]   = Ntt
    else:
        T0All        = np.concatenate((T0All, np.ones(Ntt)*T0), axis=0)

        yTemp        = np.concatenate((tout[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = np.concatenate((yMat, yTemp), axis=0)
        
        ySourceTemp  = np.concatenate((tout[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = np.concatenate((SourceMat, ySourceTemp), axis=0) 
        
        iStart[iSim] = iEnd[iSim-1]
        iEnd[iSim]   = iEnd[iSim-1]+Ntt
        

    ### Writing Results
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    # FileName = OutputDir+'/orig_data/States.csv.'+str(iSim+1)
    # np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')


    ### Writing Results
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/orig_data/y.csv.'+str(iSim+1)
    np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    FileName = OutputDir+'/orig_data/ySource.csv.'+str(iSim+1)
    np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iSim+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')


    ### Moving to New Scenario
    iSim+=1


FileName = OutputDir+'/orig_data/SimIdxs.csv'
Header   = 'iStart,iEnd'
np.savetxt(FileName, np.concatenate((iStart[...,np.newaxis], iEnd[...,np.newaxis]), axis=1), delimiter=',', header=Header, comments='')


# # Plot the Results
# iSimVec   = range(10)#[0,49,99]
# SpecOIVec = ['O2','HCO','CO','H2O','OH','CH4']

# for iSim in iSimVec:

#     for SpecOI in SpecOIVec:

#         jStart  = int(iStart[iSim])
#         jEnd    = int(iEnd[iSim])
#         for iSpec in range(gas.n_species):
#             if (gas.species_name(iSpec) == SpecOI):
#                 jSpec = iSpec
#                 break

#         fig = plt.figure(figsize=(16,12))
#         L1  = plt.plot(DataMat[jStart:jEnd,0], DataMat[jStart:jEnd,1], color='r', label='T', lw=2)
#         plt.xlabel('time (s)')
#         plt.ylabel('Temperature (K)')
#         plt.twinx()
#         L2  = plt.plot(DataMat[jStart:jEnd,0], DataMat[jStart:jEnd,iSpec+2], label=SpecOI, lw=2)
#         plt.ylabel('Mass Fraction')
#         plt.legend(L1+L2, [line.get_label() for line in L1+L2], loc='lower right')
#         plt.xscale('log')
#         FigPath = FigDir+SpecOI+'_Sim'+str(iSim+1)+'.png'
#         fig.savefig(FigPath, dpi=600)

# ##########################################################################################

