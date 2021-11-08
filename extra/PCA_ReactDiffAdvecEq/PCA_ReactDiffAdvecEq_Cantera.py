import os
import sys

import numpy  as np
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

# tMin     = 1.e-2
# tMax     = 5.e-1
# dt0      = 1.e-2
# tStratch = 1.0

# rtol     = 1.e-4
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

# tMin     = 1.e-2
# tMax     = 5.e-1
# dt0      = 1.e-2
# tStratch = 1.0

# rtol     = 1.e-4
# #####################################################################################################

#####################################################################################################
### INPUT DATA
OutDir   = WORKSPACE_PATH + '/ROMNet/Data/ReactDiffAdvecEq/'
FigDir   = OutDir + '/Figs/'

T0       = 300.
P0       = 10. * ct.one_atm
T        = 1000.

nx       = 100   # number of collocation points
Lx       = 1.0   # domain size 

tMin     = 1.e-4
tMax     = 5.e-1
dt0      = 1.e-6
tStratch = 1.1

rtol     = 1.e-4
#####################################################################################################



# #####################################################################################################
# ### RHS Function
# #def ReacDiffAdvRHS_x(x, y, dx, xx, ns, DiffFun, AdvecFun, ReacFun, gas, T, rho0):
# def ReacDiffAdvRHS_x(x, y):
#     print(x)
    
#     y         = np.append(y, [0]*nx, axis=0)
#     Y         = y.reshape(ns,nx)

#     dYdt_Diff = np.zeros((ns,nx))
#     Y_Adv     = np.zeros((ns,nx))
#     for iX in range(-1,nx-1):
#         DiffM           = DiffFun(xx[iX],  DiffC1, DiffC2)
#         dYdt_Diff[:,iX] = np.matmul(DiffM, (Y[:,iX+1] - Y[:,iX-1]) / (2.*dx) )
#         AdvecM          = AdvecFun(xx[iX], AdvecC1, AdvecC2)
#         Y_Adv[:,iX]     = np.matmul(AdvecM, Y[:,iX])
        
#     dYdt      = np.zeros((ns-1,nx))    
#     for iX in range(-1,nx-1):
#         Diff       = (dYdt_Diff[:,iX+1] - dYdt_Diff[:,iX-1]) / (2.*dx)
#         Advec      = (    Y_Adv[:,iX+1] -     Y_Adv[:,iX-1]) / (2.*dx)
       
#         gas.TDY    = T, rho0, Y[:,iX]
#         wdot       = gas.net_production_rates
#         Reac       = wdot * gas.molecular_weights / rho0   
        
#         dYdt[:,iX] = Diff[0:-1] + Advec[0:-1] + Reac[0:-1]
    
#     dydt = dYdt.flatten()
        
#     return dydt
# #####################################################################################################



#####################################################################################################
### RHS Function
#def ReacDiffAdvRHS_x(x, y, dx, xx, ns, DiffFun, AdvecFun, ReacFun, gas, T, rho0):
def ReacDiffAdvRHS_x(x, y):
    #print(x)

    Y         = y.reshape(nx,ns-1)
    Y         = np.append(Y, np.zeros((nx,1)), axis=1)

    dYdt_Diff = np.zeros((nx,ns))
    Y_Adv     = np.zeros((nx,ns))
    for iX in range(-1,nx-1):
        DiffM           = DiffFun(xx[iX],  DiffC1, DiffC2)
        dYdt_Diff[iX,:] = np.matmul(DiffM, (Y[iX+1,:] - Y[iX-1,:]) / (2.*dx) )
        AdvecM          = AdvecFun(xx[iX], AdvecC1, AdvecC2)
        Y_Adv[iX,:]     = np.matmul(AdvecM, Y[iX,:])
        
    dYdt      = np.zeros((nx,ns-1))    
    for iX in range(-1,nx-1):
        Diff       = (dYdt_Diff[iX+1,:] - dYdt_Diff[iX-1,:]) / (2.*dx)
        Advec      = (    Y_Adv[iX+1,:] -     Y_Adv[iX-1,:]) / (2.*dx)
       	
        gas.TDY    = T, rho0, Y[iX,:]
        wdot       = gas.net_production_rates
        Reac       = wdot * gas.molecular_weights / rho0   
        
        dYdt[iX,:] = Diff[0:-1] + Advec[0:-1] #+ Reac[0:-1]

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
y0  = np.tile(gas.Y[0:-1], nx)

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

#plt.savefig(FigDir+'/DiffAdvect.png', dpi=900)



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



# ---------------------------------------------------------------------------------------------------
### Write Results
DF = pd.DataFrame(np.concatenate((np.repeat(np.array(tVec), nx)[...,np.newaxis], yMat.T), axis=1), columns=['t']+gas.species_names)
DF.to_csv(OutDir+'/OrigData.csv', index=False)



# ---------------------------------------------------------------------------------------------------
### Plot Results
fig = plt.figure(figsize=(16, 12))

jt = 0
iS = 0
plt.plot(x[0+iS*nx:nx+iS*nx], yMat[iS,0+jt*nx:nx+jt*nx],  'k-', label='t = '+'{:.2f}'.format(tVec[jt])+'s')  
for iS in range(1,ns-1):
    plt.plot(x[0+iS*nx:nx+iS*nx], yMat[iS,0+jt*nx:nx+jt*nx],  'k-')  

jt = Nt-1
plt.plot(x[0+iS*nx:nx+iS*nx],yMat[iS,0+jt*nx:nx+jt*nx],  'r-', label='t = '+'{:.2e}'.format(tVec[jt])+'s') 
for iS in range(1,ns-1):
    plt.plot(x[0+iS*nx:nx+iS*nx],yMat[iS,0+jt*nx:nx+jt*nx],  'r-') 

plt.xlabel('x', fontsize=24)
plt.ylabel(r'$\theta_i$', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid()
plt.legend(fontsize=20)

plt.savefig(FigDir+'/yEnd.png', dpi=900)