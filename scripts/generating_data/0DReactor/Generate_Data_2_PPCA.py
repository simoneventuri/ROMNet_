import sys
print(sys.version)
import os
import numpy   as np
import pandas  as pd
import shutil

from PCAfold import PCA as PCAA

# WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
WORKSPACE_PATH = os.getcwd()+'/../../../../../'

# import matplotlib.pyplot as plt
# plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')



########################################################################################## 
### Input Data
###

OutputDir             = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2_NoNoise/'
#OutputDir             = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4_/'

NVarsRed              = 8
CleanVars_FilePath    = OutputDir+'/Orig/CleanVars_ToRed.csv'
NotCleanVars_FilePath = OutputDir+'/Orig/CleanVars_NotToRed.csv'

scale                 = 'log10'
MinVal                = 1.e-30

# ## FIRST TIME
# DirName               = 'train'
# n_ics                 = 500

# SECOND TIME
DirName              = 'test'
n_ics                = 10

iSimVec              = range(n_ics)

ReadFlg              = False
ReadDir              = None #WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2/'
##########################################################################################




def sample_hidden_given_visible(W_ml, C, Var_ml, X):

    q = W_ml.shape[1]
    M = (np.transpose(W_ml) @ W_ml + Var_ml * np.eye(q)).real
    
    Cov        = Var_ml * np.linalg.inv(M)
    X_pca      = []
    X_pca_mean = []
    for x in X:
        Mean       = np.linalg.inv(M) @ np.transpose(W_ml) @ (x - C)
        sample     = np.random.multivariate_normal(Mean, Cov, size=1000)
        X_pca_mean.append(Mean)
        X_pca.append(sample.T)
    
    return np.array(X_pca_mean), Cov, np.array(X_pca)



def sample_visible_given_hidden(W_ml, C, Var_ml, X_pca):
    
    n_data    = X_pca.shape[0]
    q         = X_pca.shape[1]
    n_samples = X_pca.shape[2]
    d         = W_ml.shape[0]

    Cov       = Var_ml * np.eye(d)

    X_        = np.zeros((n_data, d, X_pca.shape[2]))
    for i_data in range(n_data):
        for i_sample in range(n_samples):
            Mean                  = W_ml @ X_pca[i_data,:,i_sample] + C
            X_[i_data,:,i_sample] = np.random.multivariate_normal(Mean, Cov, size=1)
    
    return X_




### Creating Folders
try:
    os.makedirs(FigDir)
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PPC/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PPC/'+DirName+'/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PPC/ROM/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PPC/'+DirName+'/ext/')
except:
    pass

try:
    shutil.copyfile(CleanVars_FilePath,    OutputDir+'/' + str(NVarsRed) + 'PPC/CleanVars_ToRed.csv')
except:
    pass
try:
    shutil.copyfile(NotCleanVars_FilePath, OutputDir+'/' + str(NVarsRed) + 'PPC/CleanVars_NotToRed.csv')
except:
    pass




jSim    = 0
yMatCSV = []
for iSim in iSimVec:

    try:
        FileName     = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iSim+1) 
        Datay        = pd.read_csv(FileName, header=0)
        OrigVarNames = list(Datay.columns.array)[1:]
        yMatCSV.append(Datay)
        jSim+=1

    except:
        print('\n\n[PCA] File ', OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iSim+1) , ' Not Found!')
yMatCSV = pd.concat(yMatCSV, axis=0).to_numpy()




### Removing Constant Features
tOrig        = yMatCSV[:,0]
FileName = OutputDir+'/Orig/'+DirName+'/ext/t.csv'
np.savetxt(FileName, tOrig, delimiter=',')

yMatTemp     = np.maximum(yMatCSV[:,1:], MinVal)



print('\n\n[PCA] Original (', len(OrigVarNames), ') Variables: ', OrigVarNames, '\n')

try:
    KeptVarsNames_ = pd.read_csv(CleanVars_FilePath, header=None).to_numpy('str')[0,:]
except:
    KeptVarsNames_ = pd.read_csv(OutputDir+'/Orig/train/ext/CleanVars.csv', header=None).to_numpy('str')[0,:]
    print('[PCA]    REDUCING ALL VARIABLES!\n')
try:
    NotVarsNames_  = pd.read_csv(NotCleanVars_FilePath, header=None).to_numpy('str')[0,:]
except:
    NotVarsNames_  = []
print('[PCA] To Be Reduced   (', len(KeptVarsNames_), ') Species: ', KeptVarsNames_, '\n')
print('[PCA] To Be Preserved (', len(NotVarsNames_),  ') Species: ', NotVarsNames_,   '\n')


jSpec         = 0
jSpecNot      = 0
KeptVarsNames = []
NotVarsNames  = []
for iCol in range(yMatTemp.shape[1]):
    iVar    = iCol
    OrigVar = OrigVarNames[iVar]
    
    #if (np.amax(np.abs(yMatTemp[1:,iCol] - yMatTemp[:-1,iCol])) > 1.e-8):
    if (OrigVar in KeptVarsNames_):
        if (jSpec == 0):
            yMatOrig     = yMatTemp[:,iCol][...,np.newaxis]
            if   (scale == 'lin'):
                yMat     = yMatTemp[:,iCol][...,np.newaxis]
            elif (scale == 'log'):
                yMat     = np.log(yMatTemp[:,iCol][...,np.newaxis])
            elif (scale == 'log10'):
                yMat     = np.log10(yMatTemp[:,iCol][...,np.newaxis])
            # ySource      = ySourceTemp[:,iCol][...,np.newaxis]
        else:
            yMatOrig     = np.concatenate((yMatOrig, yMatTemp[:,iCol][...,np.newaxis]), axis=1)
            if   (scale == 'lin'):
                yMat     = np.concatenate((yMat,        yMatTemp[:,iCol][...,np.newaxis]), axis=1)
            elif (scale == 'log'):
                yMat     = np.concatenate((yMat, np.log(yMatTemp[:,iCol])[...,np.newaxis]), axis=1)
            elif (scale == 'log10'):
                yMat     = np.concatenate((yMat, np.log10(yMatTemp[:,iCol])[...,np.newaxis]), axis=1)
            # ySource  = np.concatenate((ySource, ySourceTemp[:,iCol][...,np.newaxis]), axis=1)
        KeptVarsNames.append(OrigVar)
        jSpec += 1


    elif (OrigVar in NotVarsNames_):
        if (jSpecNot == 0):
            yMatOrigNot  = yMatTemp[:,iCol][...,np.newaxis]
            if   (scale == 'lin'):
                yMatNot  = yMatTemp[:,iCol][...,np.newaxis]
            elif (scale == 'log'):
                yMatNot  = np.log(yMatTemp[:,iCol])[...,np.newaxis]
            elif (scale == 'log10'):
                yMatNot  = np.log10(yMatTemp[:,iCol])[...,np.newaxis]
        else:
            yMatOrigNot  = np.concatenate((yMatOrigNot, yMatTemp[:,iCol][...,np.newaxis]), axis=1)
            if   (scale == 'lin'):
                yMatNot  = np.concatenate((yMatNot,        yMatTemp[:,iCol][...,np.newaxis]), axis=1)
            elif (scale == 'log'):
                yMatNot  = np.concatenate((yMatNot, np.log(yMatTemp[:,iCol])[...,np.newaxis]), axis=1)
            elif (scale == 'log10'):
                yMatNot  = np.concatenate((yMatNot, np.log10(yMatTemp[:,iCol])[...,np.newaxis]), axis=1)
        NotVarsNames.append(OrigVar)
        jSpecNot += 1
        

#if (DirName == 'train'):
ToOrig = []
for Var in NotVarsNames:
    ToOrig.append(OrigVarNames.index(Var))
for Var in KeptVarsNames:
    ToOrig.append(OrigVarNames.index(Var))
ToOrig = np.array(ToOrig, dtype=int)

if (DirName == 'train'):
    FileName = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/ToOrig_Mask.csv'
    np.savetxt(FileName, ToOrig, delimiter=',')


### Removing Constant Features
tOrig    = yMatCSV[:,0]
FileName = OutputDir+'/Orig/'+DirName+'/ext/yCleaned.csv'
Header = 't'
for Var in KeptVarsNames:
    Header += ','+Var
np.savetxt(FileName, np.concatenate((tOrig[...,np.newaxis], yMat), axis=1), delimiter=',', header=Header)



### 
if (DirName == 'train') and (ReadFlg == False):

    NVars      = yMat.shape[1]

    C          = yMat.mean(axis=0)
    D          = yMat.std(axis=0)

    yMat_Norm  = (yMat - C)/D

    yCov       = np.cov(yMat_Norm,rowvar=False)
    L, U       = np.linalg.eig(yCov)
    Idx        = L.argsort()[::-1]   
    L          = L[Idx]
    U          = - U[:,Idx]

    Var        = (1.0 / (NVars-NVarsRed+1.e-10)) * sum([L[j] for j in range(NVarsRed,NVars)])
    Uq         = U[:,:NVarsRed]
    Lq         = np.diag(L[:NVarsRed])
    W          = ( Uq @ np.sqrt(Lq - Var * np.eye(NVarsRed)) ).real


    FileName    = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/W.csv'
    np.savetxt(FileName, W, delimiter=',')
    
    FileName    = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/V.csv'
    np.savetxt(FileName, Var[np.newaxis,...], delimiter=',')

    FileName    = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/C.csv'
    np.savetxt(FileName, C, delimiter=',')

    FileName    = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/D.csv'
    np.savetxt(FileName, D, delimiter=',')


else:
    if (ReadFlg == False):
        ReadDir = OutputDir
    FileName = ReadDir+'/'+str(NVarsRed)+'PPC/ROM/W.csv'
    W        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    WT       = W.T

    FileName = ReadDir+'/'+str(NVarsRed)+'PPC/ROM/C.csv'
    C        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    C        = np.squeeze(C)

    FileName = ReadDir+'/'+str(NVarsRed)+'PPC/ROM/D.csv'
    D        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    D        = np.squeeze(D)

    FileName = ReadDir+'/'+str(NVarsRed)+'PPC/ROM/V.csv'
    Var      = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    Var      = np.squeeze(Var)
    print('[PCA] W                 = ', W)
    print('[PCA] C                 = ', C)
    print('[PCA] D                 = ', D)
    print('[PCA] Var               = ', Var)
    print('[PCA] Shape of W        = ', W.shape)
    print('[PCA] ')


    yMat_Norm  = (yMat - C)/D



M        = (np.transpose(W) @ W + Var * np.eye(NVarsRed)).real
Cov      = Var * np.linalg.inv(M)



Header   = ''
for Var in NotVarsNames:
    Header += Var+','
Header  += 'PC_1'
for iVarsRed in range(1,NVarsRed):
    Header += ','+'PC_'+str(iVarsRed+1)

if (DirName == 'train'):
    FileName = OutputDir+'/'+str(NVarsRed)+'PPC/ROM/RedVars.csv'
    with open(FileName, 'w') as the_file:
        the_file.write(Header+'\n')



yMat_pca    = (np.linalg.inv(M) @ np.transpose(W) @ (yMat_Norm.T)).T
FileName    = OutputDir+'/' + str(NVarsRed) + 'PPC/'+DirName+'/ext/PPC.csv'
if (jSpecNot == 0):
    np.savetxt(FileName, yMat_pca, delimiter=',', header=Header, comments='')
else:
    np.savetxt(FileName, np.concatenate([yMatNot, yMat_pca], axis=1), delimiter=',', header=Header, comments='')


yMat_      = (W @ yMat_pca.T).T * D + C

print('[PCA] Shape of yMat_pca = ', yMat_pca.shape)
if   (scale == 'lin'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - yMat_)))
elif (scale == 'log'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - np.exp(yMat_))))
elif (scale == 'log10'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - 10**(yMat_))))



Header0 = Header
Header  = 't,'+Header

for iT in range(1,n_ics+1):

    try:
        FileName    = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iT) 
        Datay       = pd.read_csv(FileName, header=0)
        tVec        = Datay['t'].to_numpy()[...,np.newaxis]



        if   (scale == 'lin'):
            yTemp       = np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal)
            yNot        = np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal)
        elif (scale == 'log'):
            yTemp       = np.log(np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal))
            yNot        = np.log(np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal))
        elif (scale == 'log10'):
            yTemp       = np.log10(np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal))
            yNot        = np.log10(np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal))

        yTemp_Norm     = ((yTemp - C)/D)
        yMat_pca       = ( np.linalg.inv(M) @ np.transpose(W) @ (yTemp_Norm.T) ).T
        

        FileName    = OutputDir+'/' + str(NVarsRed) + 'PPC/'+DirName+'/ext/PPC.csv.'+str(iT)
        Temp        = np.concatenate((tVec, yNot, yMat_pca), axis=1)
        np.savetxt(FileName, Temp, delimiter=',', header=Header, comments='')

    except:
        pass
