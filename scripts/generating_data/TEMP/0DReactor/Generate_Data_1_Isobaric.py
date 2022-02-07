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
from scipy.interpolate import interp1d


##########################################################################################
### Input Data

OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_2000Cases_NEq/'
FigDir             = OutputDir + '/fig/'

MixtureFile        = 'gri30.yaml'

P0                 = ct.one_atm
DirName            = 'train'
NICs               = 50
T0Exts             = np.array([1000., 2000.], dtype=np.float64)
X0Exts             = np.array([0.05, 0.95], dtype=np.float64)
SpeciesVec         = ['H2','H','O','O2','OH','N','NH','NO','N2']
EqRatio0Exts       = None #np.array([.5, 4.], dtype=np.float64)

# DirName            = 'test'
# NICs               = 5

NPerT0             = 500

Integration        = ' '#'Canteras'
rtol               = 1.e-14
atol               = 1.e-18
SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

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

    gas_.TPY = T, P_, Y
    
    wdot     = gas_.net_production_rates

    Tdot     = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    Ydot     = wdot * gas_.molecular_weights / gas_.density

    HR       = - np.dot(gas_.net_production_rates, gas_.partial_molar_enthalpies)

    return Tdot, Ydot, HR


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


if (DirName == 'train'):

    if (EqRatio0Exts):
        MinVals = np.array([EqRatio0Exts[0], T0Exts[0]], dtype=np.float64)
        MaxVals = np.array([EqRatio0Exts[1], T0Exts[1]], dtype=np.float64)
        NDims   = 2

        ICs     = pyDOE.lhs(2, samples=NICs, criterion='center')

        for i in range(NDims):
            ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]
        ICs = np.concatenate([P0*np.ones((NICs,1)),ICs], axis=1)

        ### Writing Initial Temperatures
        FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
        Header   = 'P,EqRatio,T'
        np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')


    elif (X0Exts is not None) and (SpeciesVec is not None):
        NSpecies = len(SpeciesVec)

        MinVals  = np.array([T0Exts[0], X0Exts[0]], dtype=np.float64)
        MaxVals  = np.array([T0Exts[1], X0Exts[1]], dtype=np.float64)
        NDims    = NSpecies + 1

        ICs      = pyDOE.lhs(NDims, samples=NICs, criterion='center')

        ICs[:,0] = ICs[:,0] * (T0Exts[1] - T0Exts[0]) + T0Exts[0]
        for i in range(1, NDims):
            ICs[:,i] = ICs[:,i] * (X0Exts[1] - X0Exts[0]) + X0Exts[0]
        ICs = np.concatenate([P0*np.ones((NICs,1)),ICs], axis=1)

        ### Writing Initial Temperatures
        FileName   = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
        SpeciesStr = SpeciesVec[0]
        for Spec in SpeciesVec[1:]:
            SpeciesStr += ','+Spec
        Header   = 'P,T,'+SpeciesStr
        np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')

    else:
        print('Please, specify (EqRatio0Exts) OR (X0Exts and SpeciesVec)!')



elif (DirName == 'test'):
    NDims    = 2
    ICs      = np.zeros((NICs,NDims))
    # ICs[:,0] = [2.5, 1.9, 3.5, 1., 3.6]
    # ICs[:,1] = [1200., 1900., 1300., 1600., 1700.]
    ICs[:,0] = [0.8, 0.9, 1.0, 1.1, 1.2]
    ICs[:,1] = [1300., 1200., 1400., 1500., 1250.]
    ICs = np.concatenate([P0*np.ones((NICs,1)), ICs], axis=1)


    ### Writing Initial Temperatures
    FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
    Header   = 'P,EqRatio,T'
    np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')



### Iterating Over Residence Times
DataMat         = None
iStart          = np.zeros(NICs)
iEnd            = np.zeros(NICs)
AutoIgnitionVec = np.zeros((NICs,1))
for iIC in range(NICs):
    
    if (EqRatio0Exts):
        P0       = ICs[iIC,0]
        EqRatio0 = ICs[iIC,1]
        T0       = ICs[iIC,2]
        print('Pressure = ', P0, 'Pa; EqRatio0 = ', EqRatio0, '; Temperature = ', T0, 'K')

    elif (X0Exts is not None) and (SpeciesVec is not None):
        P0       = ICs[iIC,0]
        T0       = ICs[iIC,1]
        print('Pressure = ', P0, 'Pa; Temperature = ', T0, 'K')
    

    ### Create Mixture
    gas     = ct.Solution(MixtureFile)

    ### Create Reactor
    gas.TP  = T0, P0

    if (EqRatio0Exts):
        # gas.set_equivalence_ratio(EqRatio0, 'CH4:1.0', 'O2:0.21, N2:0.79')
        # gas.set_equivalence_ratio(EqRatio0, 'CH4:1.0', 'O2:1.0')
        gas.set_equivalence_ratio(EqRatio0, 'H2:1.0', 'O2:1.0, N2:4.0')

    elif (X0Exts is not None) and (SpeciesVec is not None):
        SpecDict = {}
        for iS, Spec in enumerate(SpeciesVec):
            SpecDict[Spec] = ICs[iIC,iS+2]
        gas.X    = SpecDict
        print('   Mole Fractions = ', SpecDict)

    r       = ct.IdealGasConstPressureReactor(gas)
    sim     = ct.ReactorNet([r])
    sim.verbose = False

    gas_    = gas
    mass_   = r.mass
    print('   Mass = ', mass_)
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
    TVec  = np.array([700, 800, 900, 1000, 1200, 1500, 1700, 1850, 2000])
    tVec1 = np.array([5.e1, 5.e0, 1.e-1, 1.e-4, 1.e-5, 5.e-6, 1.e-6, 5.e-7, 5.e-7])
    tVec2 = np.array([5.e0, 1.e0, 1.e-2, 1.e-5, 1.e-6, 5.e-7, 1.e-7, 5.e-8, 5.e-8])
    tVec3 = np.array([1.e4, 1.e2, 1.e0, 1.e-1, 5.e-2, 1.e-2, 1.e-1, 5.e-2, 1.e-2])

    f1 = interp1d(1000/TVec, np.log10(tVec1), kind='cubic')
    f2 = interp1d(1000/TVec, np.log10(tVec2), kind='cubic')
    f3 = interp1d(1000/TVec, np.log10(tVec3), kind='cubic')

    tMin     = f1(1000/T0) #1.e-5
    dt0      = f2(1000/T0) #1.e-5
    tMax     = f3(1000/T0) #1.e-3
    tMaxAll  = f1(1.) #1.e-5

    # tStratch = 1.3
    # tVec     = [0.0]
    # t        = 1.e-12 #10**tMin
    # dt0      = 1.e-6  #10**dt0
    # dt       = dt0
    # while (t <= 1.e-2):#10**tMax):
    #     tVec.append(t)
    #     t  =   t + dt
    #     dt = dt0 * tStratch
    # print(len(tVec))
    # tVec     = np.concatenate([[0.], np.logspace(tMin, tMax, 3000)])
    #tVec     = np.concatenate([[0.], np.logspace(-12, tMin, 20), np.logspace(tMin, tMax, 480)[1:]])
    #tVec     = np.concatenate([[0.], np.logspace(-12, -6, 100), np.logspace(-5.99999999, -1., 4899)])
    tVec     = np.concatenate([np.logspace(-12., 0., NPerT0)])
    #tVec     = np.concatenate([[0.], np.linspace(1.e-12, 1.e-2, 4999)])
    #############################################################################

    


    gas_             = gas
    states           = ct.SolutionArray(gas, 1, extra={'t': [0.0]})
    
    if (Integration == 'Canteras'):
        TT               = r.T
        YY               = r.thermo.Y
        Vec              = np.concatenate(([TT],YY), axis=0)
        TTdot, YYdot, HR = IdealGasConstPressureReactor(tVec[0], TT, YY)
        Vecdot           = np.concatenate(([TTdot],YYdot), axis=0)
        Mat              = np.array(Vec[np.newaxis,...])
        Source           = np.array(Vecdot[np.newaxis,...])
        it0              = 1
        tVecFinal        = np.array(tVec, dtype=np.float64)
        HRVec            = [HR]
    else:
        output           = solve_ivp( IdealGasConstPressureReactor_SciPY, (tVec[0],tVec[-1]), y0, method=SOLVER, t_eval=tVec, atol=atol )
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
            YY               = output.y[1:,it]
            # YY               = np.concatenate((output.y[1:,it], [1.0-np.sum(output.y[1:,it])]), axis=0)

        Vec                  = np.concatenate(([TT],YY), axis=0)

        TTdot, YYdot, HR     = IdealGasConstPressureReactor(t, TT, YY)
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

    FileName = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iIC+1)
    np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    FileName = OutputDir+'/Orig/'+DirName+'/ext/ySource.csv.'+str(iIC+1)
    np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iIC+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')



FileName = OutputDir+'/Orig/'+DirName+'/ext/SimIdxs.csv'
Header   = 'iStart,iEnd'
np.savetxt(FileName, np.concatenate((iStart[...,np.newaxis], iEnd[...,np.newaxis]), axis=1), delimiter=',', header=Header, comments='')

FileName = OutputDir+'/Orig/'+DirName+'/ext/tAutoIgnition.csv'
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

