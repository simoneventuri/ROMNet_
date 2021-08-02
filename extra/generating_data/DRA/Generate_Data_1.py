import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
WORKSPACE_PATH  = os.environ['WORKSPACE_PATH']
plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

from scipy.integrate import solve_ivp


##########################################################################################
### Input Data

OutputDir          = WORKSPACE_PATH + '/ROMNet/Data_1DRA_Clean/'
FigDir             = OutputDir + '/fig/'

tStratch           = 1.
Nt                 = 10
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




#RHS function
def DiffReacAdvODE(x, y):
    nx   = int(len(y)/ns)
    dydt = np.zeros(y.shape[0])
    
    Y = np.zeros((ns, nx))
    for iS in range(ns):
        Y[iS,:] = y[0+iS*nx:nx+iS*nx]

    dYdt_Diff = np.zeros((ns,nx))
    Y_Adv     = np.zeros((ns,nx))
    for iX in range(-1,nx-1):
        DiffM           = DiffFun(xx[iX], ns)
        dYdt_Diff[:,iX] = np.matmul(DiffM, (Y[:,iX+1] - Y[:,iX-1]) / (2.*dx) )
        AdvecM          = AdvecFun(xx[iX], ns)
        Y_Adv[:,iX]     = np.matmul(AdvecM, Y[:,iX])
        
    dYdt  = np.zeros((ns,nx))
    for iX in range(-1,nx-1):
        Diff       = (dYdt_Diff[:,iX+1] - dYdt_Diff[:,iX-1]) / (2.*dx)
        Advec      = (    Y_Adv[:,iX+1] -     Y_Adv[:,iX-1]) / (2.*dx)
        
        #ReacM = ReacFun(xx[iX], ns)
        #Reac  = np.matmul(ReacM, Y[:,iX])
        
#         #KInel, KDiss, KRec = ReacFun(xx[iX], ns)
#         Reac       = Diff*0.0
#         Reac[0:-1] = (- KDiss*Y[0:-1,iX] + KRec*Y[-1,iX]**2)*Y[-1,iX] #+ (np.matmul(KInelT, Y[0:-1,iX]))*Y[-1,iX]
#         Reac[-1]   = np.sum( (KDiss*Y[0:-1,iX] - KRec*Y[-1,iX]**2)*Y[-1,iX] )
        
        Reac       = (np.matmul(KInelT, Y[:,iX]))
        
        dYdt[:,iX] = Diff + Advec + Reac
        
    
    dydt = dYdt.flatten()
        
    return dydt





#DIFFUSION-REACTION OF MULTIPLE SCALARS

ns = 50    # number of scalars
nx = 300   # number of collocation points
Lx = 1.0   # domain size 

dx = Lx/nx           # grid spacing
X  = np.zeros((ns,nx))
x  = np.zeros(nx*ns) 
for iS in range(0,ns):
    x[0+iS*nx:nx+iS*nx] = np.linspace(0., Lx-dx, nx)
    X[iS,:]             = np.linspace(0., Lx-dx, nx)
xx = x[0:nx]

#initial profile
sig = 0.05       
y0  = 1.0/(sig*np.sqrt(2.*np.pi)) * np.exp( -0.5*( (x-0.5 )/sig )**2.  )
#y0  = np.piecewise(x, [np.abs(x-0.5) < 0.05, np.abs(x-0.5) >= 0.05], [10, 0])


Y0 = np.zeros((ns,nx))
for iS in range(ns):
    Y0[iS,:] = 1.0 / ( sig*np.sqrt(2.*np.pi)) * np.exp( -0.5*( (X[iS,:]-0.5 )/sig )**2. )
    #Y0[iS,:] = np.piecewise(X[iS,:], [np.abs(X[iS,:]-0.5) < 0.05, np.abs(X[iS,:]-0.5) >= 0.05], [10, 0])
