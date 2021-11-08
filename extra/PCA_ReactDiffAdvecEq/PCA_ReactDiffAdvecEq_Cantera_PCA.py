import os
import sys

import numpy as np
import pandas as pd

import matplotlib
from matplotlib        import pyplot as plt
from matplotlib        import cm
from matplotlib.ticker import LinearLocator

WORKSPACE_PATH  = os.environ['WORKSPACE_PATH']
plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

import cantera         as ct

from scipy.integrate   import solve_ivp, ode



# #####################################################################################################
# ### INPUT DATA
# OutDir   = WORKSPACE_PATH + '/ROMNet/Data/DiffAdvecEq/'
# FigDir   = OutDir + '/Figs/'

# T0       = 300.
# P0       = 10. * ct.one_atm
# T        = 1000.

# nx       = 100   # number of collocation points
# Lx       = 1.0   # domain size 

# IntegFlg = False
# tMin     = 1.e-2
# tMax     = 5.e-1
# dt0      = 1.e-2
# tStratch = 1.0

# rtol     = 1.e-4

# ns_Red   = 3
# #####################################################################################################

# #####################################################################################################
# ### INPUT DATA
# OutDir   = WORKSPACE_PATH + '/ROMNet/Data/ReacEq/'
# FigDir   = OutDir + '/Figs/'

# T0       = 300.
# P0       = 10. * ct.one_atm
# T        = 1000.

# nx       = 100   # number of collocation points
# Lx       = 1.0   # domain size 

# IntegFlg = False
# tMin     = 1.e-2
# tMax     = 5.e-1
# dt0      = 1.e-2
# tStratch = 1.0

# rtol     = 1.e-4

# ns_Red     = 10
# #####################################################################################################

#####################################################################################################
### INPUT DATA
OutDir   = WORKSPACE_PATH + '/ROMNet/Data/ReactDiffAdvecEq/'
FigDir   = OutDir + '/Figs/'

T0       = 300.
P0       = 10. * ct.one_atm
T        = 1000.

nx       = 100    # number of collocation points
Lx       = 1.0   # domain size 

IntegFlg = False
tMin     = 1.e-4
tMax     = 5.e-1
dt0      = 1.e-6
tStratch = 1.1

rtol     = 1.e-4

ns_Red   = 10
#####################################################################################################



#####################################################################################################
### RHS Function
def ReacDiffAdvRHS_PCA_x(x, y):
    #print(x)
    
    #ns_       = ns
    #ns_       = nsClean
    ns_       = ns_Red

    Y         = y.reshape(nx,ns_)

    dYdt_Diff = np.zeros((nx,ns_))
    Y_Adv     = np.zeros((nx,ns_))
    for iX in range(-1,nx-1):
        DiffM_Red       = DiffFun(xx[iX],  DiffC1Clean, DiffC2Clean)
        DiffM_Red       = np.matmul( APCA, np.matmul(DiffM_Red, APCA.T) )
        dYdt_Diff[iX,:] = np.matmul(DiffM_Red, (Y[iX+1,:] - Y[iX-1,:]) / (2.*dx) )
        AdvecM_Red      = AdvecFun(xx[iX], AdvecC1Clean, AdvecC2Clean)
        AdvecM_Red      = np.matmul( APCA, np.matmul(AdvecM_Red, APCA.T) )
        Y_Adv[iX,:]     = np.matmul(AdvecM_Red, Y[iX,:])
        
    dYdt      = np.zeros((nx,ns_))    
    for iX in range(-1,nx-1):
        Diff       = (dYdt_Diff[iX+1,:] - dYdt_Diff[iX-1,:]) / (2.*dx)
        Advec      = (    Y_Adv[iX+1,:] -     Y_Adv[iX-1,:]) / (2.*dx)
        
        YClean           = pca.reconstruct(Y[iX,:][np.newaxis,...], nocenter=False)
        YOrig            = np.zeros(ns) 
        for iSClean in range(nsClean):
            YOrig[MaskClean[iSClean]] = YClean[0,iSClean]
        gas.TDY          = T, rho0, YOrig
        wdot             = gas.net_production_rates
        ReacOrig         = wdot * gas.molecular_weights / rho0   
        #ReacClean        = np.zeros(nsClean)
        ReacClean        = ReacOrig[MaskClean]
        Reac             = pca.transform(ReacClean[np.newaxis,:], nocenter=True)

        dYdt[iX,:] = Diff + Advec #+ Reac

    dydt = dYdt.flatten() 

    #sys.exit('Enough')
    return dydt
#####################################################################################################



try:
    os.makedirs(OutDir)
except OSError as e:
    pass
try:
    os.makedirs(FigDir)
except OSError as e:
    pass



# ---------------------------------------------------------------------------------------------------
### Create Mixture
gas      = ct.Solution('gri30.yaml', energy_enabled=False)
gas.TPX  = T0, P0, 'CH4:0.5, O2:1, N2:3.76'
rho0     = gas.density
ns       = gas.n_species

yOrigNames = gas.species_names





# ---------------------------------------------------------------------------------------------------
### Create Spatial Grid
dx = Lx/nx         
X  = np.zeros((ns,nx))
x  = np.zeros(nx*ns) 
for iS in range(0,ns):
    x[0+iS*nx:nx+iS*nx] = np.linspace(0., Lx-dx, nx)
    X[iS,:]             = np.linspace(0., Lx-dx, nx)
xx = x[0:nx]



# ---------------------------------------------------------------------------------------------------
### Create Initial Profile and Plot It

#sig = 0.05       
#y0  = 1.0/(sig*np.sqrt(2.*np.pi)) * np.exp( -0.5*( (x-Lx/2)/sig )**2.  ) * np.repeat([gas.Y], nx)
y0    = np.tile(gas.Y[0:-1], nx)
y0All = np.tile(gas.Y, nx)


Y0 = np.zeros((ns,nx))
for iS in range(ns):
    #Y0[iS,:] = 1.0 / ( sig*np.sqrt(2.*np.pi)) * np.exp( -0.5*( (X[iS,:]-Lx/2)/sig )**2. ) * gas.Y[iS]    
    Y0[iS,:] = np.ones(nx) * gas.Y[iS]    

fig = plt.figure(figsize=(16, 12))
plt.plot(X.T, Y0.T)
plt.xlabel('x')
plt.ylabel(r'$\theta_i$ at t=0 [s]')

#plt.savefig(FigDir+'/y0.png', dpi=900)



# ---------------------------------------------------------------------------------------------------
### Create Diffusion and Advection Properties and Plotting Them

np.random.seed(3)
DiffC1 = 0.01*np.random.rand(ns) 
DiffC2 = 2.*np.pi*np.random.rand(ns)
def DiffFun(x, DiffC1, DiffC2):
    DiffV  = DiffC1 * np.cos(x*(2.*np.pi)+DiffC2) + 0.03
    DiffM  = np.diag(DiffV)
    return DiffM


np.random.seed(4)
AdvecC1 = 2.*np.random.rand(ns)
AdvecC2 = 2.*np.pi*np.random.rand(ns)
def AdvecFun(x, AdvecC1, AdvecC2):
    AdvecV  = AdvecC1 * np.sin(x*(2.*np.pi) + AdvecC2)
    AdvecM  = np.diag(AdvecV)
    return AdvecM


NTemp      = 100
DiffMPlot  = np.zeros((ns,NTemp))
AdvecMPlot = np.zeros((ns,NTemp))
ix         = 0
xTemp      = np.linspace(0.,1.,NTemp)
for ix in range(NTemp):
    DiffMPlot[:,ix]  = np.diag(DiffFun(xTemp[ix], DiffC1, DiffC2))
    AdvecMPlot[:,ix] = np.diag(AdvecFun(xTemp[ix], AdvecC1, AdvecC2))
    ix+=1


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
Slice = list(range(0,50,10))

ax1.plot(xTemp, DiffMPlot[Slice].T)
ax1.legend(['Species '+str(int(x)+1) for x in Slice], fontsize=20)
ax1.set_xlabel('x', fontsize=24)
ax1.set_ylabel(r'$D_i(x)$', fontsize=24)
#ax1.set_xticks(fontsize=20)
#ax1.set_yticks(fontsize=20)
ax1.grid()

ax2.plot(xTemp, AdvecMPlot[Slice].T)
ax2.legend(['Species '+str(int(x)+1) for x in Slice], fontsize=20)
ax2.set_xlabel('x', fontsize=24)
ax2.set_ylabel(r'$A_i(x)$', fontsize=24)
#ax2.set_xticks(fontsize=20)
#ax2.set_yticks(fontsize=20)
ax2.grid()

#plt.savefig(FigDir+'/Diff_Advec.png', dpi=900)



if (IntegFlg):


    # ---------------------------------------------------------------------------------------------------
    ### Create Time Grid
    tVec     = [0.0]
    t        = tMin
    dt       = dt0
    while (t <= tMax):
        tVec.append(t)
        t   = t + dt
        dt *= tStratch
    tRef    = tMin
    Nt = len(tVec)



    # ---------------------------------------------------------------------------------------------------
    ### Initialize Matrixes
    xMat  = np.zeros((ns,nx))
    for i in range(0,ns):
        xMat[i,:] =  x[0+i*nx:nx+i*nx]



    # ---------------------------------------------------------------------------------------------------
    ### Integrate and Unpack
    # dYdt      = np.zeros((ns-1,nx))
    # dYdt_Diff = np.zeros((ns,nx))
    # Y_Adv     = np.zeros((ns,nx))

    output    = solve_ivp( ReacDiffAdvRHS_x, (tVec[0],tVec[-1]), y0, method='BDF')

    y_next    = []
    Nt        = output.y.shape[1]
    for it in range(Nt):
        y_next.append(output.y[:,it])

    yTemp = np.zeros((ns,nx))
    for it in range(Nt):
        for ix in range(0,nx):
            yTemp[0:ns-1,ix] = y_next[it][ix*(ns-1):(ix+1)*(ns-1)]
        yTemp[ns-1,:] = 1.0 - np.sum(yTemp, axis=0)
        if (it == 0):
            yMat = yTemp
        else:
            yMat = np.concatenate((yMat,yTemp), axis=1)

    tVec = output.t

else:
    
    DF   = pd.read_csv(OutDir+'/OrigData.csv')
    tVec = np.unique(DF['t'].to_numpy())
    yMat = DF[gas.species_names].to_numpy().T
    Nt   = int(yMat.shape[1]/nx)




# ---------------------------------------------------------------------------------------------------
### Clean
KeepOrig   = list(np.sum(yMat.T, axis=0) > 1.e-10) # list(yMat.std(axis=1) > 1.e-8)
yCleanNames= [yOrigNames[i] for i in range(len(yOrigNames)) if KeepOrig[i]]
yPDClean   = DF[yCleanNames]
yClean     = yPDClean.to_numpy()
nsClean    = yClean.shape[1]
print('nsClean = ', nsClean)

Y0Clean    = np.zeros((nsClean,nx))
iSClean    = 0
MaskClean  = np.zeros(nsClean, dtype=int)
CleanMask  = np.zeros(ns, dtype=int)
for iS in range(ns):
    if KeepOrig[iS]:
        # Y0Clean[iSClean,,:] = 1.0 / ( sig*np.sqrt(2.*np.pi)) * np.exp( -0.5*( (X[iS,:]-Lx/2)/sig )**2. ) * gas.Y[iS]    
        Y0Clean[iSClean,:]  = np.ones(nx) * Y0[iS,:]
        MaskClean[iSClean]  = iS
        CleanMask[iS]       = iSClean
        iSClean            += 1
        
y0Clean = Y0Clean.T.flatten()        

fig = plt.figure(figsize=(16, 12))
plt.plot(xx, Y0Clean.T)
plt.xlabel('x')
plt.ylabel(r'$\theta_i$ at t=0 [s]')



# ---------------------------------------------------------------------------------------------------
### Reduce
yMean           = np.zeros((nsClean,1))#pca.mean_[...,np.newaxis]

from PCAfold import PCA as PCAA
pca        = PCAA(yClean, scaling='pareto', n_components=ns_Red)
yMat_pca   = pca.transform(yClean, nocenter=False)
C          = pca.X_center
D          = pca.X_scale
A          = pca.A[:,0:ns_Red].T
L          = pca.L
AT         = A.T
# yMat_pca   = ((yMat.T - C)/D).dot(A)
yMat_      = pca.reconstruct(yMat_pca, nocenter=False)

## For Verification: 
print('Shape of yMat_pca = ', yMat_pca.shape)
print('Shape of A        = ', A.shape)
print('Error = ', np.max(abs(yClean - yMat_)))



# ---------------------------------------------------------------------------------------------------
### Reduced Quantities

X_Red  = X
Y0_Red = pca.transform(Y0Clean.T, nocenter=True).T
y0_Red = np.zeros((ns_Red*nx))
for i in range(0,ns_Red):
    y0_Red[0+i*nx:nx+i*nx] = Y0_Red[i,:]

    
y0_Red = Y0_Red.T.flatten()
    
## For Verification: 
Y0_    = pca.reconstruct(Y0_Red.T, nocenter=True).T
print(np.max(abs(Y0Clean-Y0_)))


DiffC1Clean  = DiffC1[MaskClean]
DiffC2Clean  = DiffC2[MaskClean]

AdvecC1Clean = AdvecC1[MaskClean]
AdvecC2Clean = AdvecC2[MaskClean]

APCA         = A




# ---------------------------------------------------------------------------------------------------
### Integrate and Unpack

#y0_       = y0All
#y0_       = y0Clean
y0_       = y0_Red
output    = solve_ivp( ReacDiffAdvRHS_PCA_x, (tVec[0],tVec[-1]), y0_, method='BDF')




y_next    = []
Nt        = output.y.shape[1]
for it in range(Nt):
    y_next.append(output.y[:,it])

ns_   = int(output.y.shape[0]/nx)
yTemp = np.zeros((ns_,nx))
for it in range(Nt):
    for ix in range(0,nx):
        yTemp[0:ns_,ix] = y_next[it][ix*(ns_):(ix+1)*(ns_)]
    if (it == 0):
        yMat_Red = yTemp
    else:
        yMat_Red = np.concatenate((yMat_Red,yTemp), axis=1)

tVecRed = output.t



# ---------------------------------------------------------------------------------------------------
### Write Results
PCNames = ['PC_1']
for iPC in range(1,ns_Red):
    PCNames.append('PC_'+str(iPC+1))

DF = pd.DataFrame(np.concatenate((np.repeat(np.array(tVecRed), nx)[...,np.newaxis], yMat_Red.T), axis=1), columns=['t']+PCNames)
DF.to_csv(OutDir+'/PCData.csv', index=False)





ColorVec = ['k','r','b','y','g','m','c']*10


fig = plt.figure(figsize=(16, 12))

nt    = int(yMat_Red.shape[1]/nx)
it    = 9
for i in range(0,ns_Red):
    plt.plot(x[0+i*nx:nx+i*nx],yMat_Red[i,0+it*nx:nx+it*nx], ColorVec[i]+'-', label=r'$\eta_{'+str(i+1)+'}$')  #solutions at time t=t1

plt.xlabel('x', fontsize=24)
plt.ylabel(r'$\eta_i$', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.legend(fontsize=20)
#plt.show()
plt.savefig(FigDir+'/PCEnd_ROM.png', dpi=900)




ColorVec = ['k','r','b','y','g','m','c']*10

yMat_Red.T

yMat_ = pca.reconstruct(yMat_Red.T, nocenter=True).T

# for i in range(0,ns):
#     plt.plot(x[0+i*nx:nx+i*nx],y0[0+i*nx:nx+i*nx], 'b-')  #initial condition
fig = plt.figure(figsize=(16, 12))

ntClean = int(yClean.T.shape[1]/nx)
itClean = ntClean-1
nt      = int(yMat_.shape[1]/nx)
it      = nt-1
for i in range(0,nsClean):
    plt.plot(x[0+i*nx:nx+i*nx],yClean.T[i,0+itClean*nx:nx+itClean*nx], ColorVec[i]+'-', label=r'$y_{'+str(i+1)+'}$')  #solutions at time t=t1
    plt.plot(x[0+i*nx:nx+i*nx],yMat_[i,0+it*nx:nx+it*nx], ColorVec[i]+':')  #solutions at time t=t1

plt.xlabel('x', fontsize=24)
plt.ylabel(r'$y_i$', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.legend(fontsize=20)
#plt.show()

plt.savefig(FigDir+'/yEnd_ROM.png', dpi=900)