import sys
print(sys.version)
import os
import numpy   as np
import pandas  as pd

import matplotlib.pyplot as plt
WORKSPACE_PATH  = os.environ['WORKSPACE_PATH']
plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')

from PCAfold import PCA as PCAA



##########################################################################################
### Input Data
###
OutputDir          = WORKSPACE_PATH + '/ROMNet/Data_PSR_Clean_Test/'
TrainDir           = WORKSPACE_PATH + '/ROMNet/Data_10PSR_Clean_Lin/'
FigDir             = OutputDir + '/fig/'

NRests             = 1
iSimVec            = range(NRests)

NVarsRed           = 3
DirName            = '/orig_data/'
##########################################################################################




### Creating Folders
try:
    os.makedirs(FigDir)
except:
    pass
try:
    os.makedirs(OutputDir+'/pca_' + str(NVarsRed) + '/')
except:
    pass
try:
    os.makedirs(OutputDir+'/pc_data_' + str(NVarsRed) + '/')
except:
    pass



### Retrieving Data
FileName    = OutputDir+DirName+'/ResidenceTimes.csv'
Data        = pd.read_csv(FileName, header=None)
RestVecOrig = np.squeeze(Data.to_numpy())
try:
    NSimOrig = len(RestVecOrig)
    RestVec  = RestVecOrig[iSimVec]
except:
    NSimOrig = 1
    RestVec  = np.array([RestVecOrig])
print('[PCA] Found    Residence Times Vector: ', RestVecOrig)
print('[PCA] Required Residence Times Vector: ', RestVec)
print('[PCA] ')



FileName         = TrainDir+'/orig_data/CleanVars.csv'
VarsName         = list(pd.read_csv(FileName, delimiter=',', header=0).columns)
KeptSpeciesNames = VarsName
print('[PCA] Final (', len(KeptSpeciesNames), ') Variables: ', KeptSpeciesNames)
print('[PCA] ')



jSim=0
for iSim in iSimVec:
    FileName     = OutputDir+DirName+'/y.csv.'+str(iSim+1) 
    Datay        = pd.read_csv(FileName, header=0)
    print(Datay.head())
    SpeciesNames = list(Datay.columns.array)[1:]
    FileName     = OutputDir+DirName+'/ySource.csv.'+str(iSim+1) 
    DataS        = pd.read_csv(FileName, header=0)
    if (jSim == 0):
        tOrig        = Datay['t'].to_numpy()
        yMat         = Datay[VarsName].to_numpy()
        ySource      = DataS[VarsName].to_numpy()
        RestVecTot   = np.ones((Datay.shape[0],1))*RestVec[iSim]
    else:
        tOrig        = np.concatenate((tOrig,      Datay['t'].to_numpy()), axis=0)
        yMat         = np.concatenate((yMat,       Datay[VarsName].to_numpy()), axis=0)
        ySource      = np.concatenate((ySource,    DataS[VarsName].to_numpy()), axis=0)
        RestVecTot   = np.concatenate((RestVecTot, np.ones((Datay.shape[0],1))*RestVec[iSim]), axis=0)
    jSim+=1




### Removing Constant Features
FileName = OutputDir+DirName+'/t.csv'
np.savetxt(FileName, tOrig, delimiter=',')



### Removing Constant Features
FileName = OutputDir+DirName+'/yCleaned.csv'
Header = 't'
for Var in KeptSpeciesNames:
    Header += ','+Var
np.savetxt(FileName, np.concatenate((tOrig[...,np.newaxis], yMat), axis=1), delimiter=',', header=Header)

FileName = OutputDir+DirName+'/RestVecTot.csv'
np.savetxt(FileName, RestVecTot, delimiter=',')



### 
FileName = TrainDir+'/pca_'+str(NVarsRed)+'/A.csv'
A        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
AT       = A.T

FileName = TrainDir+'/pca_'+str(NVarsRed)+'/C.csv'
C        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
C        = np.squeeze(C)

FileName = TrainDir+'/pca_'+str(NVarsRed)+'/D.csv'
D        = pd.read_csv(FileName, delimiter=',', header=None).to_numpy()
D        = np.squeeze(D)



Header   = 'PC_1'
for iVarsRed in range(1,NVarsRed):
    Header += ','+'PC_'+str(iVarsRed+1)
HeaderS  = 'SPC_1'
for iVarsRed in range(1,NVarsRed):
    HeaderS += ','+'SPC_'+str(iVarsRed+1)

#yMat_pca    = pca.transform(yMat, nocenter=False)
yMat_pca   = ((yMat - C)/D).dot(AT)
FileName    = OutputDir+'/pc_data_'+str(NVarsRed)+'/PC.csv'
np.savetxt(FileName, yMat_pca, delimiter=',', header=Header, comments='')

#ySource_pca = pca.transform(ySource, nocenter=False)
ySource_pca = (ySource/D).dot(AT) 
FileName    = OutputDir+'/pc_data_'+str(NVarsRed)+'/PCSource.csv'
np.savetxt(FileName, ySource_pca, delimiter=',', header=HeaderS, comments='')

FileName    = OutputDir+'/pc_data_'+str(NVarsRed)+'/PCAll.csv'
Temp        = np.concatenate((yMat_pca, ySource_pca), axis=1)
np.savetxt(FileName, Temp, delimiter=',', header=Header+','+HeaderS, comments='')

# For Verification: 
#yMat_      = pca.reconstruct(yMat_pca, nocenter=False)
yMat_      = (yMat_pca.dot(A))*D + C
print('[PCA] Shape of yMat_pca = ', yMat_pca.shape)
print('[PCA] Error = ', np.max(abs(yMat - yMat_)))

#ySource_      = ySource_pca.dot(A)*D 
ySource_      = (ySource_pca.dot(A))*D 
print('[PCA] Shape of ySource_pca = ', ySource_pca.shape)
print('[PCA] Error = ', np.max(abs(ySource - ySource_)))
print('[PCA] ')


Header  = 't,'+Header
HeaderS = 't,'+HeaderS

for iRest in range(1,NRests+1):
    print(KeptSpeciesNames)

    FileName    = OutputDir+DirName+'/y.csv.'+str(iRest) 
    Datay       = pd.read_csv(FileName, header=0)
    tVec        = Datay['t'].to_numpy()[...,np.newaxis]
    yTemp       = Datay[KeptSpeciesNames].to_numpy()
    #print('yTemp = ', yTemp)


    yMat_pca    = ((yTemp - C)/D).dot(AT)
    FileName    = OutputDir+'/pc_data_'+str(NVarsRed)+'/PC.csv.'+str(iRest)
    Temp        = np.concatenate((tVec, yMat_pca), axis=1)
    np.savetxt(FileName, Temp, delimiter=',', header=Header, comments='')

    FileName    = OutputDir+DirName+'/ySource.csv.'+str(iRest) 
    Datay       = pd.read_csv(FileName, header=0)
    tVec        = Datay['t'].to_numpy()[...,np.newaxis]
    ySourceTemp = Datay[KeptSpeciesNames].to_numpy()

    ySource_pca = (ySourceTemp/D).dot(AT) 
    FileName    = OutputDir+'/pc_data_'+str(NVarsRed)+'/PCSource.csv.'+str(iRest)
    Temp        = np.concatenate((tVec, ySource_pca), axis=1)
    np.savetxt(FileName, Temp, delimiter=',', header=HeaderS, comments='')
