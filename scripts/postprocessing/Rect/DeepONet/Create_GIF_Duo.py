### Importing Libraries

import sys
print(sys.version)
import os
import time


### Defining WORKSPACE_PATH

# WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
WORKSPACE_PATH = os.path.join(os.getcwd(), '../../../../../../')
ROMNet_fld     = os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/')

### Importing External Libraries

import numpy                             as np
import pandas                            as pd


### Importing Matplotlib and Its Style

import matplotlib.pyplot                 as plt
import matplotlib.cm                     as cm
import matplotlib.animation              as animation
from matplotlib import gridspec

#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/presentation.mplstyle'))
#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/zoomed.mplstyle'))
plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/paper_1column.mplstyle'))
#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/paper_2columns.mplstyle'))


from scipy.integrate import solve_ivp
import pyDOE
from PCAfold         import PCA          as PCAA


import romnet                            as rmnt_1
import romnet                            as rmnt_2

import importlib
import importlib.util

#######################################################################################################################
###

path_to_run_fld_1 = os.path.join(WORKSPACE_PATH, 'ROMNet/Rect_100Instants_TransRotScale_Rand/DeepONet/TestCase1/')
path_to_run_fld_2 = os.path.join(WORKSPACE_PATH, 'ROMNet/Rect_100Instants_TransRotScale/DeepONet/Run_18/')
RotFlg      = True
TransFlg    = True
ScaleFlg    = True

FigDir          = os.path.join(WORKSPACE_PATH, '../Desktop/Paper_Figures_DeepONet_TEMP/')


TrainingCases   = [0]#[0,2,4,6,8]
TestCases       = [0,2,4]#[0,2]

NSamples        = 1

LineVec         = ['-',':','--','.-']*10
ColorVec        = ['#190707', '#dd3232', '#0065a9', '#348a00','#985396','#f68b69']




#######################################################################################################################
###

Theta0  = 0./180*np.pi
w_Theta = 36./180*np.pi

w_Psi = 50./180*np.pi
ca    = 0.5
cx0   = 1.
cy0   = -0.5

lx0   = 8.
ly0   = 6.

xMin    = -15.
xMax    = 15.
yMin    = -15.
yMax    = 15.

v_zz  = 0.2
zz0   = 1.




#######################################################################################################################
###

spec  = importlib.util.spec_from_file_location("inputdata", path_to_run_fld_1+'./ROMNet_Input.py')
foo_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo_1)
InputData_1 = foo_1.inputdata(WORKSPACE_PATH)

InputData_1.InputFilePath    = path_to_run_fld_1+'/ROMNet_Input.py'
InputData_1.train_int_flg    = 0
InputData_1.path_to_run_fld  = path_to_run_fld_1

surrogate_type               = InputData_1.surrogate_type
Net_1                        = getattr(rmnt_1.nn, surrogate_type)

model_1                      = rmnt_1.model.Model_TF(InputData_1)

if (InputData_1.phys_system is not None):
    System_1 = getattr(rmnt_1.pinn.system, InputData_1.phys_system)
    system_1 = System_1(InputData_1)
    
model_1.build(InputData_1, None, Net_1, system_1)#, loadfile_no='000027')





#######################################################################################################################
###

spec  = importlib.util.spec_from_file_location("inputdata", path_to_run_fld_2+'./ROMNet_Input.py')
foo_2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo_2)
InputData_2 = foo_2.inputdata(WORKSPACE_PATH)

InputData_2.InputFilePath    = path_to_run_fld_2+'/ROMNet_Input.py'
InputData_2.train_int_flg    = 0
InputData_2.path_to_run_fld  = path_to_run_fld_2

surrogate_type               = InputData_2.surrogate_type
Net_2                        = getattr(rmnt_2.nn, surrogate_type)

model_2                      = rmnt_2.model.Model_TF(InputData_2)

if (InputData_2.phys_system is not None):
    System_2 = getattr(rmnt_2.pinn.system, InputData_2.phys_system)
    system_2 = System_2(InputData_2)
    
model_2.build(InputData_2, None, Net_2, system_2)#, loadfile_no='000027')






#--------------------------------------------------------------------------------------
input_vars  = model_1.net.input_vars
trunk_vars  = InputData_1.input_vars['DeepONet']['Trunk']
branch_vars = InputData_1.input_vars['DeepONet']['Branch']



plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/video.mplstyle'))

img     = [] # some array of images
frames  = [] # for storing the generated images
# fig     = plt.figure(figsize=(20,8))
# axs     = []
# axs.append( fig.add_subplot(1,3,1) )
# axs.append( fig.add_subplot(1,3,2) )
# axs.append( fig.add_subplot(1,3,3) )
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22,8))
axs                  = [ax1, ax2, ax3]
#fig.tight_layout()
#nrow = 1
#ncol = 3
#gs   = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0, top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 


Nx      = 200
Ny      = 200

t0      = 0.
tEnd    = 10.
Nt      = 100

xMin_   = -15.
xMax_   = 15.
yMin_   = -15.
yMax_   = 15.

# = 10.
axx = 10.
bxx = 10.
ayy = 10.
byy = 10.
    
    
tVec    = np.linspace(t0,tEnd,Nt)
for it, t in enumerate(tVec):
    print('t=',t)
    
    
    if (RotFlg == True):
        #Theta = Theta0 + w_Theta*t
        Theta = Theta0 + np.cos(t/10.*(4*np.pi)+2.)*np.pi
    else:
        Theta = Theta0

    if (TransFlg == True):
        Psi   = t  * w_Psi
        cr    = ca * Psi
        cx    = cx0 + cr * np.cos(Psi)
        cy    = cy0 + cr * np.sin(Psi)  
    else:
        cx    = 0.
        cy    = 0.

    if (ScaleFlg == True):
        s     = (np.sin(t/tEnd*360. / 180.*np.pi)+2)
    else:
        s     = 1.

    zz    = zz0 + t*v_zz


    Mat       = np.zeros((Nx,Ny))
    InputPred = np.zeros((Nx*Ny,3))
    
    x    = np.linspace(xMin,xMax,Nx)
    y    = np.linspace(yMin,yMax,Ny)

    i    = 0
    for ix, x_ in enumerate(x):
        for iy, y_ in enumerate(y):
            InputPred[i,0] = t
            InputPred[i,1] = x_
            InputPred[i,2] = y_
            
            xrot_          = x_*np.cos(Theta) - y_*np.sin(Theta)
            yrot_          = x_*np.sin(Theta) + y_*np.cos(Theta)
            xrot_          = xrot_-cx
            yrot_          = yrot_-cy
            xrot_          = xrot_*s
            yrot_          = yrot_*s
            zx_1           = (np.tanh((xrot_+lx0)*axx) + np.tanh((lx0-xrot_)*bxx))/2
            zy_1           = (np.tanh((yrot_+ly0)*ayy) + np.tanh((ly0-yrot_)*byy))/2

            Mat[ix,iy]     = np.exp(zx_1+zy_1)*zz
            i+=1
            
    InputPred = pd.DataFrame(InputPred, columns=['t','x','y'])
    yMat_1    = model_1.predict(InputPred)
    yMat_2    = model_2.predict(InputPred)
    
    #axs.append(plt.subplot(gs[0,0]))
    im1       = axs[0].imshow((Mat).reshape(Nx,Ny), animated=True, origin='lower', cmap=cm.turbo, extent=([xMin, xMax, yMin, yMax]))
    axs[0].set_title("Rigid Body Dynamics", pad=10, fontsize=26)
    axs[0].set_xlabel('x', fontsize=24)
    axs[0].set_ylabel('y', fontsize=24)

    #axs.append(plt.subplot(gs[0,1]))
    im2       = axs[1].imshow((yMat_1).reshape(Nx,Ny), animated=True, origin='lower', cmap=cm.turbo, extent=([xMin, xMax, yMin, yMax]))
    rect      = plt.Rectangle((-10.,-10.), 20., 20., linewidth=2, edgecolor=ColorVec[0], facecolor='none')
    axs[1].add_patch(rect)
    axs[1].text( -9.5,  8.2, r'Test Predictions', color=ColorVec[0], fontsize=26, weight='bold')
    axs[1].text(-14.5, 13.2, r'Extrapolations', color=ColorVec[0], fontsize=26, weight='bold')
    axs[1].set_title("Vanilla DeepONet architecture,\n 165,761 trainable parameters", pad=10, fontsize=26)
    axs[1].set_xlabel('x', fontsize=24)
    axs[1].set_yticklabels([])

    #axs.append(plt.subplot(gs[0,2]))
    im3       = axs[2].imshow((yMat_2).reshape(Nx,Ny), animated=True, origin='lower', cmap=cm.turbo, extent=([xMin, xMax, yMin, yMax]))
    rect      = plt.Rectangle((-10.,-10.), 20., 20., linewidth=2, edgecolor=ColorVec[1], facecolor='none')
    axs[2].add_patch(rect)
    axs[2].text( -9.5,  8.2, r'Test Predictions', color=ColorVec[1], fontsize=26, weight='bold')
    axs[2].text(-14.5, 13.2, r'Extrapolations', color=ColorVec[1], fontsize=26, weight='bold')
    axs[2].set_title("Proposed DeepONet extension,\n 1,921 trainable parameters", pad=10, fontsize=26)
    axs[2].set_xlabel('x', fontsize=24)
    axs[2].set_yticklabels([])
    
    frames.append([im1,im2,im3])
    
ani = animation.ArtistAnimation(fig, frames, interval=5000, blit=True, repeat_delay=50000)
writergif = animation.PillowWriter(fps=30)
ani.save(path_to_run_fld_2+'/Figures/Video.gif',writer=writergif)