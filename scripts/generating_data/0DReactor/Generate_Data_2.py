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

OutputDir             = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2/'
#OutputDir             = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4_/'

NVarsRed              = 15
CleanVars_FilePath    = OutputDir+'/Orig/CleanVars_ToRed.csv'
NotCleanVars_FilePath = OutputDir+'/Orig/CleanVars_NotToRed.csv'

scale                 = 'log10'
MinVal                = 1.e-30

# ## FIRST TIME
# DirName               = 'train'
# n_ics                 = 500

# SECOND TIME
DirName               = 'test'
n_ics                 = 10

iSimVec               = range(n_ics)

ReadFlg               = False
ReadDir               = None #WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_H2/'
##########################################################################################



### Creating Folders
try:
    os.makedirs(FigDir)
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/ROM/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/')
except:
    pass

try:
    shutil.copyfile(CleanVars_FilePath,    OutputDir+'/' + str(NVarsRed) + 'PC/CleanVars_ToRed.csv')
except:
    pass
try:
    shutil.copyfile(NotCleanVars_FilePath, OutputDir+'/' + str(NVarsRed) + 'PC/CleanVars_NotToRed.csv')
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
# ySourceTemp  = SourceMatCSV[:,1:]



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
        

if (DirName == 'train'):
    ToOrig = []
    for Var in NotVarsNames:
        ToOrig.append(OrigVarNames.index(Var))
    for Var in KeptVarsNames:
        ToOrig.append(OrigVarNames.index(Var))
    ToOrig = np.array(ToOrig, dtype=int)

    FileName = OutputDir+'/'+str(NVarsRed)+'PC/ToOrig_Mask.csv'
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

    pca        = PCAA(yMat, scaling='std', n_components=NVarsRed, nocenter=False)
    C          = pca.X_center
    D          = pca.X_scale
    A          = pca.A[:,0:NVarsRed].T
    L          = pca.L
    AT         = A.T
    print('[PCA] Shape of A        = ', A.shape)
    print('[PCA] ')

    FileName    = OutputDir+'/'+str(NVarsRed)+'PC/ROM/A.csv'
    np.savetxt(FileName, A, delimiter=',')

    FileName    = OutputDir+'/'+str(NVarsRed)+'PC/ROM/C.csv'
    np.savetxt(FileName, C, delimiter=',')

    FileName    = OutputDir+'/'+str(NVarsRed)+'PC/ROM/D.csv'
    np.savetxt(FileName, D, delimiter=',')

else:
    if (ReadFlg == False):
        ReadDir = OutputDir
    FileName = ReadDir+'/'+str(NVarsRed)+'PC/ROM/A.csv'
    A        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    AT       = A.T

    FileName = ReadDir+'/'+str(NVarsRed)+'PC/ROM/C.csv'
    C        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    C        = np.squeeze(C)

    FileName = ReadDir+'/'+str(NVarsRed)+'PC/ROM/D.csv'
    D        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
    D        = np.squeeze(D)
    print('[PCA] A                 = ', A)
    print('[PCA] C                 = ', C)
    print('[PCA] D                 = ', D)
    print('[PCA] Shape of A        = ', A.shape)
    print('[PCA] ')



Header   = ''
for Var in NotVarsNames:
    Header += Var+','
Header  += 'PC_1'
for iVarsRed in range(1,NVarsRed):
    Header += ','+'PC_'+str(iVarsRed+1)

# HeaderS  = 'SPC_1'
# for iVarsRed in range(1,NVarsRed):
#     HeaderS += ','+'SPC_'+str(iVarsRed+1)

if (DirName == 'train'):
    FileName = OutputDir+'/'+str(NVarsRed)+'PC/ROM/RedVars.csv'
    with open(FileName, 'w') as the_file:
        the_file.write(Header+'\n')


#yMat_pca    = pca.transform(yMat, nocenter=False)
yMat_pca    = ((yMat - C)/D).dot(AT)
FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/PC.csv'
if (jSpecNot == 0):
    np.savetxt(FileName, yMat_pca, delimiter=',', header=Header, comments='')
else:
    np.savetxt(FileName, np.concatenate([yMatNot, yMat_pca], axis=1), delimiter=',', header=Header, comments='')

# #ySource_pca = pca.transform(ySource, nocenter=False)
# ySource_pca = (ySource/D).dot(AT) 
# FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/PCSource.csv'
# np.savetxt(FileName, ySource_pca, delimiter=',', header=HeaderS, comments='')

# FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/PCAll.csv'
# Temp        = np.concatenate((yMat_pca, ySource_pca), axis=1)
# np.savetxt(FileName, Temp, delimiter=',', header=Header+','+HeaderS, comments='')

# For Verification: 
#yMat_      = pca.reconstruct(yMat_pca, nocenter=False)
yMat_      = (yMat_pca.dot(A))*D + C
print('[PCA] Shape of yMat_pca = ', yMat_pca.shape)
if   (scale == 'lin'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - yMat_)))
elif (scale == 'log'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - np.exp(yMat_))))
elif (scale == 'log10'):
    print('[PCA] Error = ', np.max(abs(yMatOrig - 10**(yMat_))))

# #ySource_      = ySource_pca.dot(A)*D 
# ySource_      = (ySource_pca.dot(A))*D 
# print('[PCA] Shape of ySource_pca = ', ySource_pca.shape)
# print('[PCA] Error = ', np.max(abs(ySource - ySource_)))
# print('[PCA] ')



Header0 = Header
Header  = 't,'+Header
# HeaderS = 't,'+HeaderS

# fDeepOnetInput  = open(OutputDir +'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/Input.csv', 'w')
# fDeepOnetOutput = open(OutputDir +'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/Output.csv', 'w')

for iT in range(1,n_ics+1):

    try:
        FileName    = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iT) 
        Datay       = pd.read_csv(FileName, header=0)
        tVec        = Datay['t'].to_numpy()[...,np.newaxis]


        # FileName    = OutputDir+'/Orig/'+DirName+'/ext/ySource.csv.'+str(iT) 
        # Datay       = pd.read_csv(FileName, header=0)
        # tVec        = Datay['t'].to_numpy()[...,np.newaxis]
        # ySourceTemp = np.maximum(Datay[KeptVarsNames].to_numpy(), 0.)


        if   (scale == 'lin'):
            yTemp       = np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal)
            yNot        = np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal)
        elif (scale == 'log'):
            yTemp       = np.log(np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal))
            yNot        = np.log(np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal))
        elif (scale == 'log10'):
            yTemp       = np.log10(np.maximum(Datay[KeptVarsNames].to_numpy(), MinVal))
            yNot        = np.log10(np.maximum(Datay[NotVarsNames].to_numpy(),  MinVal))

        yMat_pca       = ((yTemp - C)/D).dot(AT)
        # ySourceMat_pca = ((ySourceTemp)/D).dot(AT)
        

        FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/PC.csv.'+str(iT)
        Temp        = np.concatenate((tVec, yNot, yMat_pca), axis=1)
        np.savetxt(FileName, Temp, delimiter=',', header=Header, comments='')


        # FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/SPC.csv.'+str(iT)
        # Temp        = np.concatenate((tVec, ySourceMat_pca), axis=1)
        # np.savetxt(FileName, Temp, delimiter=',', header=HeaderS, comments='')


        # yMat_pca0   = np.tile(yMat_pca[0,:],(yMat_pca.shape[0],1)) 
        # Temp0        = np.concatenate((tVec, yMat_pca0), axis=1)
        # if (iT==1):
        #     np.savetxt(fDeepOnetInput,  Temp0, delimiter=',', header=Header, comments='')
        #     np.savetxt(fDeepOnetOutput, Temp,  delimiter=',', header=Header, comments='')
        # else:
        #     np.savetxt(fDeepOnetInput,  Temp0, delimiter=',')
        #     np.savetxt(fDeepOnetOutput, Temp,  delimiter=',')
    except:
        pass

        # FileName    = OutputDir+'/Orig/'+DirName+'/ext/ySource.csv.'+str(iT) 
        # Datay       = pd.read_csv(FileName, header=0)
        # tVec        = Datay['t'].to_numpy()[...,np.newaxis]
        # ySourceTemp = Datay[KeptVarsNames].to_numpy()

        # ySource_pca = (ySourceTemp/D).dot(AT) 
        # FileName    = OutputDir+'/' + str(NVarsRed) + 'PC/'+DirName+'/ext/PCSource.csv.'+str(iT)
        # Temp        = np.concatenate((tVec, ySource_pca), axis=1)
        # np.savetxt(FileName, Temp, delimiter=',', header=HeaderS, comments='')

# fDeepOnetInput.close()
# fDeepOnetOutput.close()
