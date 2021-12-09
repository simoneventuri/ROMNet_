import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time

import pyDOE

import matplotlib.pyplot as plt
WORKSPACE_PATH  = os.environ['WORKSPACE_PATH']
plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

import cantera as ct
from scipy.integrate import solve_ivp


##########################################################################################
### Input Data

OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Adiabatic_100Cases/'
FigDir             = OutputDir + '/fig/'

MixtureFile        = 'gri30.yaml'

NICs               = 100
P0Exts             = np.array([5. * ct.one_atm, 20. * ct.one_atm], dtype=np.float64)
EqRatio0Exts       = np.array([0.5, 1.0], dtype=np.float64)
T0Exts             = np.array([900, 1700], dtype=np.float64)

NPerT0             = 1000

Integration        = ' '#'Canteras'
#rtol               = 1.e-8
atol               = 1.e-9
SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

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
def IdealGasConstPressureReactor_SciPY(t, y):
    #print(t)

    YEnd     = np.array([1.-np.sum(y[1:])], dtype=np.float64)
    Y        = np.concatenate((y[1:], YEnd), axis=0)
    gas_.TPY = y[0], P_, Y
    
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / gas_.density
    
    return ydot


def IdealGasConstPressureReactor(t, T, Y):

    gas_.TPY = T, P_, Y
    
    wdot     = gas_.net_production_rates

    Tdot     = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    Ydot     = wdot * gas_.molecular_weights / gas_.density
    
    return Tdot, Ydot


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



##########################################################################################
### Generating Training Data


MinVals = np.array([P0Exts[0], EqRatio0Exts[0], T0Exts[0]], dtype=np.float64)
MaxVals = np.array([P0Exts[1], EqRatio0Exts[1], T0Exts[1]], dtype=np.float64)
NDims   = 3

ICs     = pyDOE.lhs(3, samples=100, criterion='center')

for i in range(NDims):
    ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]



### Writing Initial Temperatures
FileName = OutputDir+'/orig_data/ICs.csv'
Header   = 'P,EqRatio,T'
np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')



### Iterating Over Residence Times
DataMat         = None
iStart          = np.zeros(NICs)
iEnd            = np.zeros(NICs)
AutoIgnitionVec = np.zeros((NICs,1))
for iIC in range(NICs):
    P0       = ICs[iIC,0]
    EqRatio0 = ICs[iIC,1]
    T0       = ICs[iIC,2]
    print('Pressure = ', P0, 'Pa; EqRatio0 = ', EqRatio0, '; Temperature = ', T0, 'K')
    

    ### Create Mixture
    gas     = ct.Solution(MixtureFile)

    ### Create Reactor
    gas.TP  = T0, P0
    # gas.set_equivalence_ratio(EqRatio0, 'CH4:1.0', 'O2:0.21, N2:0.79')
    gas.set_equivalence_ratio(EqRatio0, 'CH4:1.0', 'O2:1.0')
    r       = ct.IdealGasReactor(gas)
    sim     = ct.ReactorNet([r])
    sim.verbose = False

    gas_    = gas
    mass_   = r.mass
    density_= r.density

    y0      = np.array(np.hstack((gas_.T, gas_.Y[0:-1])), dtype=np.float64)


    ### Initialize Integration 
    # tAuto    = 10**( (7.37601476*(1000/T0) - 8.44036162) - (P0/ct.one_atm-10)/10 )
    tAuto    = 10**( (8.30140399*(1000/T0) -9.9625361) - (P0/ct.one_atm-10)/10 )
    ### print('tAuto = ', tAuto)
    tMin     = tAuto * 1.e-3
    tMax     = tAuto * 5.e1
    dt0      = tMin
    tStratch = 1.01
    tVec     = [0.0]
    t        = tMin
    dt       = dt0
    while (t <= tMax):
        tVec.append(t)
        t  =   t + dt
        dt = dt0 * tStratch

    gas_             = gas
    states           = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
    
    if (Integration == 'Canteras'):
        TT               = r.T
        YY               = r.thermo.Y
        Vec              = np.concatenate(([TT],YY), axis=0)
        TTdot, YYdot, HR = IdealGasReactor(tVec[0], TT, YY)
        Vecdot           = np.concatenate(([TTdot],YYdot), axis=0)
        Mat              = np.array(Vec[np.newaxis,...])
        Source           = np.array(Vecdot[np.newaxis,...])
        it0              = 1
        tVecFinal        = np.array(tVec, dtype=np.float64)
        HRVec            = [HR]
    else:
        output           = solve_ivp( IdealGasReactor_SciPY, (tVec[0],tVec[-1]), y0, method=SOLVER, t_eval=tVec, atol=atol )
        it0              = 0
        tVecFinal        = output.t
        HRVec            = []

    ### Integrate
    it           = it0
    for t in tVecFinal[it:]:

        if (Integration == 'Canteras'):
            sim.advance(t)
            TT               = r.T
            YY               = r.thermo.Y
        else:
            TT               = output.y[0,it]
            YY               = np.concatenate((np.maximum(output.y[1:,it],0.), [1.0-np.minimum(np.sum(output.y[1:,it]),1.)]), axis=0)
        
        Vec                  = np.concatenate(([TT],YY), axis=0)

        TTdot, YYdot, HR     = IdealGasReactor(t, TT, YY)
        Vecdot               = np.concatenate(([TTdot],YYdot), axis=0)

        if (it == 0):
            Mat              = np.array(Vec[np.newaxis,...])
            Source           = np.array(Vecdot[np.newaxis,...])
        else:
            Mat              = np.concatenate((Mat, Vec[np.newaxis,...]),       axis=0)
            Source           = np.concatenate((Source, Vecdot[np.newaxis,...]), axis=0)

        HRVec.append(HR)
        it+=1 
        
    AutoIgnitionVec[iIC,0]   = tVecFinal[HRVec.index(max(HRVec))+it0]   
    ### print('Auto Ignition Delay = ', auto_ignition)


    ### Storing Results
    Nt  = len(tVecFinal)
    if (Nt < NPerT0):
        Mask = np.arange(Nt)
        Ntt  = Nt
    else:
        Mask = np.linspace(0,Nt-1,NPerT0, dtype=int)
        Ntt  = NPerT0

    if (iIC == 0):
        T0All        = np.ones(Ntt)*T0
        yTemp        = np.concatenate((tVecFinal[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = yTemp

        ySourceTemp  = np.concatenate((tVecFinal[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = ySourceTemp
        
        iStart[iIC]  = 0
        iEnd[iIC]    = Ntt
    else:
        T0All        = np.concatenate((T0All, np.ones(Ntt)*T0), axis=0)

        yTemp        = np.concatenate((tVecFinal[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = np.concatenate((yMat, yTemp), axis=0)
        
        ySourceTemp  = np.concatenate((tVecFinal[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = np.concatenate((SourceMat, ySourceTemp), axis=0) 
        
        iStart[iIC]  = iEnd[iIC-1]
        iEnd[iIC]    = iEnd[iIC-1]+Ntt
        

    ### Writing Results
    NSpec    = gas.n_species
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    # FileName = OutputDir+'/orig_data/States.csv.'+str(iIC+1)
    # np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')


    ### Writing Results
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/orig_data/y.csv.'+str(iIC+1)
    np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    FileName = OutputDir+'/orig_data/ySource.csv.'+str(iIC+1)
    np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iIC+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')



FileName = OutputDir+'/orig_data/SimIdxs.csv'
Header   = 'iStart,iEnd'
np.savetxt(FileName, np.concatenate((iStart[...,np.newaxis], iEnd[...,np.newaxis]), axis=1), delimiter=',', header=Header, comments='')

FileName = OutputDir+'/orig_data/tAutoIgnition.csv'
Header   = 't'
np.savetxt(FileName, AutoIgnitionVec, delimiter=',', header=Header, comments='')



# # Plot the Results
# iSimVec   = range(10)#[0,49,99]
# SpecOIVec = ['O2','HCO','CO','H2O','OH','CH4']

# for iIC in iSimVec:

#     for SpecOI in SpecOIVec:

#         jStart  = int(iStart[iIC])
#         jEnd    = int(iEnd[iIC])
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
#         FigPath = FigDir+SpecOI+'_Sim'+str(iIC+1)+'.png'
#         fig.savefig(FigPath, dpi=600)

# ##########################################################################################

