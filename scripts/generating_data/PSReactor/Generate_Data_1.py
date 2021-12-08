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

OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/PSR_100Cases/'
FigDir             = OutputDir + '/fig/'

MixtureFile        = 'gri30.yaml'
NRests             = 100
RestVec            = np.logspace(np.log10(1.e-5), np.log10(1.e-4), NRests) # [2.e-5]
#RestVec            = np.concatenate([np.linspace(1.e-6, 1.e-5, 20), np.linspace(2.e-5, 1.e-4, 19)])# [2.e-5]
NPerRest           = 1000

tStratch           = 1.
Nt                 = NPerRest*2

T0Inlet            = 300.
P0Inlet            = ct.one_atm
EqRatioInlet       = 1.

TrainFlg           = 'train'
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
    os.makedirs(OutputDir+'/Orig/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+TrainFlg+'/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+TrainFlg+'/ext/')
except:
    pass
OutputDir += '/Orig/'+TrainFlg+'/ext'




##########################################################################################
### Defining ODE and its Parameters

def ReactorOde_CVODE(t, y):

    mass     = np.sum(y[1:])
    gas_.HPY = y[0]/mass, P_, y[1:]/mass

    rho      = gas_.density
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y)
    ydot[0]  = (mass*hIn_ - y[0]) / Rest_
    ydot[1:] = wdot * gas_.molecular_weights * V_ + (YIn_ - y[1:]) * mdot_

    #print('t = ', t, '; ydot = ', ydot)

    return ydot


def ReactorOde_CVODE_2(t, y):
    #print(t)

    mass     = np.sum(y[1:])
    gas_.HPY = y[0]/mass, P_, y[1:]/mass

    rho      = gas_.density
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y)
    ydot[0]  = (mass*hIn_ - y[0]) / Rest_
    ydot[1:] = wdot * gas_.molecular_weights * V_ + (YIn_ - y[1:]) * mdot_

    return ydot



def ReactorOde_Jacobian(t, y):
    print(t)

    Eps = 1e-6
    J   = np.zeros([len(y), len(y)], dtype = np.float)

    for i in range(len(y)):
        y1 = y.copy()
        y2 = y.copy()

        y1[i] += Eps
        y2[i] -= Eps

        f1 = ReactorOde_CVODE_2(t, y1)
        f2 = ReactorOde_CVODE_2(t, y2)

        J[ : , i] = (f1 - f2) / (2. * Eps)

    return J



def ProduceSource(t, y):
    
    mass       = np.sum(y[1:])
    gas_.HPY   = y[0]/mass, P_, y[1:]/mass

    rho        = gas_.density
    wdot       = gas_.net_production_rates

    ydot_      = np.zeros_like(y)
    ydot_[0]   = (mass*hIn_ - y[0]) / Rest_
    ydot_[1:]  = wdot * gas_.molecular_weights * V_ + (YIn_ - y[1:]) * mdot_

    return ydot_[np.newaxis,...]

##########################################################################################





##########################################################################################
### Generating Training Data

### Writing Residence Times
FileName = OutputDir+'/ResidenceTimes.csv'
np.savetxt(FileName, RestVec)



### Iterating Over Residence Times
DataMat  = None
iStart   = np.zeros(len(RestVec))
iEnd     = np.zeros(len(RestVec))
iSim     = 0
for Rest in RestVec:
    print('Rest = ', Rest)
    

    ### Create Mixture
    gas   = ct.Solution(MixtureFile)
    cpi   = gas.partial_molar_cp/gas.molecular_weights
    NSpec = gas.n_species


    ### Create Inlet
    gas.TP      = T0Inlet, P0Inlet 
    gas.set_equivalence_ratio(EqRatioInlet, 'CH4:1.0', 'O2:1.0, N2:0.0', basis='mass')
    YIn_        = gas.Y
    hIn_        = np.dot(gas.X/gas.volume_mole, gas.partial_molar_enthalpies) / gas.density

    ### Create Combustor
    gas.equilibrate('HP')
    gas_       = gas
    P_         = gas.P
    V_         = 1.0
    h0         = np.dot(gas.X/gas.volume_mole, gas.partial_molar_enthalpies)/gas.density
    gas_.HP    = h0, gas.P
    Ny         = gas.n_species+1

    Rest_      = Rest
    mdot_      = gas.density * V_ / Rest_

    y0         = np.array(np.hstack((h0*gas.density*V_, gas.Y*gas.density*V_)), dtype=np.float64)


    ### Initialize Integration           
    dt0        = 1.e-10
    tMax       = Rest*1.e+3
    tout       = [0.]
    #tout       = np.concatenate((np.array(tout), np.logspace(np.log10(dt0), np.log10(tMax), Nt-1)), axis=0)
    tout       = np.logspace(np.log10(dt0), np.log10(tMax), Nt-1)
    states     = ct.SolutionArray(gas_, 1, extra={'t': tout[0]})
    SOLVER     = 'BDF'


    output     = solve_ivp( ReactorOde_CVODE, tout[[0,-1]], y0, method=SOLVER, t_eval=tout, atol=1.e-20, first_step=1.e-14)#, )


    ### Integrate
    NTott    = len(output.t)
    #JJTauMat = np.zeros((NTott, output.y.shape[0]))
    for it in range(NTott):
        t = output.t[it]
        u = output.y[:,it]

        mass     = np.sum(u[1:])
        gas_.HPY = u[0]/mass, P_, u[1:]/mass
        
        if (it==0):

            if (t==0.):
                Mat    = y0[np.newaxis,...]
                Source = ProduceSource(0., y0)
            else:
                Mat    = u[np.newaxis,...]
                Source = ProduceSource(t, u)

            #JJ     = ReactorOde_Jacobian(t, y0)
            

        else:
            Mat        = np.concatenate((Mat, u[np.newaxis,...]), axis=0)
            states.append(gas_.state, t=t)
            SourceTemp = ProduceSource(t, u)
            Source     = np.concatenate((Source, SourceTemp), axis=0)

            #JJ         = ReactorOde_Jacobian(t, u)
           
        # JJEig, JJVec   = np.linalg.eig(JJ)
        # JJTauMat[it,:] = 1./JJEig.real


    ### Storing Results
    Nt  = len(states.t)
    if (Nt < NPerRest):
        Mask = np.arange(Nt)
        Ntt  = Nt
    else:
        Mask = np.linspace(0,Nt-1,NPerRest, dtype=int)
        Ntt  = NPerRest

    if (iSim == 0):
        RestAll      = np.ones(Ntt)*Rest
        
        DataTemp     = np.concatenate((states.t[Mask,np.newaxis], states.T[Mask,np.newaxis], states.Y[Mask,:]), axis=1)
        DataMat      = DataTemp

        yTemp        = np.concatenate((states.t[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = yTemp

        ySourceTemp  = np.concatenate((states.t[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = ySourceTemp
        
        iStart[iSim] = 0
        iEnd[iSim]   = Ntt
    else:
        RestAll      = np.concatenate((RestAll, np.ones(Ntt)*Rest), axis=0)
        
        DataTemp     = np.concatenate((states.t[Mask,np.newaxis], states.T[Mask,np.newaxis], states.Y[Mask,:]), axis=1)
        DataMat      = np.concatenate((DataMat, DataTemp), axis=0)

        yTemp        = np.concatenate((states.t[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = np.concatenate((yMat, yTemp), axis=0)
        
        ySourceTemp  = np.concatenate((states.t[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = np.concatenate((SourceMat, ySourceTemp), axis=0) 
        
        iStart[iSim] = iEnd[iSim-1]
        iEnd[iSim]   = iEnd[iSim-1]+Ntt


    print('DataTemp = ', DataTemp)
        

    ### Writing Results
    Header   = 't,T'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/States.csv.'+str(iSim+1)
    np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')


    ### Writing Results
    Header   = 't,HH'
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/y.csv.'+str(iSim+1)
    np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    FileName = OutputDir+'/ySource.csv.'+str(iSim+1)
    np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iSim+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')


    ### Moving to New Scenario
    iSim+=1


FileName = OutputDir+'/SimIdxs.csv'
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

