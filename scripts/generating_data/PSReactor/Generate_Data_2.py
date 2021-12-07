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
OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/PSR_100Cases/'
FigDir             = OutputDir + '/fig/'

NRests             = 100
iSimVec            = range(NRests)

NVarsRed           = 3

TrainFlg           = 'train'
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
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/ROM/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/' + TrainFlg + '/')
except:
    pass
try:
    os.makedirs(OutputDir+'/' + str(NVarsRed) + 'PC/' + TrainFlg + '/ext/')
except:
    pass


### Retrieving Data
FileName    = OutputDir+'/Orig/'+TrainFlg+'/ext/ResidenceTimes.csv'
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

jSim=0
for iSim in iSimVec:
    FileName     = OutputDir+'/Orig/'+TrainFlg+'/ext/y.csv.'+str(iSim+1) 
    Datay        = pd.read_csv(FileName, header=0)
    print(Datay.head())
    SpeciesNames = list(Datay.columns.array)[1:]
    FileName     = OutputDir+'/Orig/'+TrainFlg+'/ext/ySource.csv.'+str(iSim+1) 
    DataS        = pd.read_csv(FileName, header=0)
    if (jSim == 0):
        yMatCSV      = Datay.to_numpy()
        SourceMatCSV = DataS.to_numpy()
        RestVecTot   = np.ones((Datay.shape[0],1))*RestVec[iSim]
    else:
        yMatCSV      = np.concatenate((yMatCSV,      Datay.to_numpy()), axis=0)
        SourceMatCSV = np.concatenate((SourceMatCSV, DataS.to_numpy()), axis=0)
        RestVecTot   = np.concatenate((RestVecTot,   np.ones((Datay.shape[0],1))*RestVec[iSim]), axis=0)
    jSim+=1




### Removing Constant Features
tOrig        = yMatCSV[:,0]
FileName = OutputDir+'/Orig/'+TrainFlg+'/ext/t.csv'
np.savetxt(FileName, tOrig, delimiter=',')

yMatTemp     = yMatCSV[:,1:]
ySourceTemp  = SourceMatCSV[:,1:]

yMat         = yMatTemp[:,0][...,np.newaxis]
ySource      = ySourceTemp[:,0][...,np.newaxis]
VarsName    = ['HH']
print('[PCA] Original (', len(SpeciesNames), ') Species: ', SpeciesNames)
for iSpec in range(1, yMatTemp.shape[1]):
    if (np.amax(np.abs(yMatTemp[1:,iSpec] - yMatTemp[:-1,iSpec])) > 1.e-10):
        yMat    = np.concatenate((yMat,    yMatTemp[:,iSpec][...,np.newaxis]), axis=1)
        ySource = np.concatenate((ySource, ySourceTemp[:,iSpec][...,np.newaxis]), axis=1)
        #print(gas.species_name(i-1))
        VarsName.append(SpeciesNames[iSpec])        
KeptSpeciesNames = VarsName
#print('yMat = ', yMat)
print('[PCA] Final (', len(KeptSpeciesNames), ') Variables: ', KeptSpeciesNames)
print('[PCA] ')


ToOrig = []
for Var in VarsName:
    ToOrig.append(SpeciesNames.index(Var))
ToOrig = np.array(ToOrig, dtype=int)

FileName    = OutputDir+'/'+str(NVarsRed)+'PC/ROM/ToOrig_Mask.csv'
np.savetxt(FileName, ToOrig, delimiter=',')


### Removing Constant Features
tOrig    = yMatCSV[:,0]
FileName = OutputDir+'/Orig/'+TrainFlg+'/ext/yCleaned.csv'
Header = 't'
for Var in KeptSpeciesNames:
    Header += ','+Var
np.savetxt(FileName, np.concatenate((tOrig[...,np.newaxis], yMat), axis=1), delimiter=',', header=Header)

FileName = OutputDir+'/Orig/'+TrainFlg+'/ext/RestVecTot.csv'
np.savetxt(FileName, RestVecTot, delimiter=',')

FileName = OutputDir+'/Orig/'+TrainFlg+'/ext/CleanVars.csv'
StrSep = ','
with open(FileName, 'w') as the_file:
    the_file.write(StrSep.join(VarsName)+'\n')

# ### Checking Singular Value Decomposition
# U, S, VT = np.linalg.svd(yMat, full_matrices=1)
# V        = VT.T

# fig1     = plt.figure(figsize=(16, 12))
# ax1      = fig1.add_subplot(121)
# ax1.semilogy(S,'-o',color='k')
# ax1.set_xlabel('Index', fontsize=24)
# ax1.set_ylabel('Singular Value', fontsize=24)
# #ax1.set_xticks(fontsize=20)
# #ax1.set_yticks(fontsize=20)
# ax2      = fig1.add_subplot(122)
# ax2.plot(np.cumsum(S)/np.sum(S),'-o',color='k')
# ax2.set_xlabel('Index', fontsize=24)
# ax2.set_ylabel('Energy', fontsize=24)
# #ax2.set_xticks(fontsize=20)
# #ax2.set_yticks(fontsize=20)
# #plt.show()
# FigPath = FigDir+'/SVD.png'
# fig1.savefig(FigPath, dpi=900)




# ### Checking Errors of PCA Based on Different Types of Scaling
# NVars          = yMat.shape[1]
# NVarsRedVec    = range(1, NVars)
# VarOI          = 'CH4'
# ScalingTypeVec = ['level', 'pareto', 'range', 'std', 'vast']

# for iy in range(NVars):
#     if (VarsName[iy] == VarOI):
#         jSpec = iSpec
#         break

# RMSE_Mat        = np.zeros((len(ScalingTypeVec),len(NVarsRedVec)))
# RMSE_Source_Mat = np.zeros((len(ScalingTypeVec),len(NVarsRedVec)))

# fig = plt.figure(figsize=(16,12))
# i   = 0
# for ScalingType in ScalingTypeVec:
    
#     j=0
#     for NVarsRedTemp in NVarsRedVec:

#         pca        = PCAA(yMat, scaling=ScalingType, n_components=NVarsRedTemp)
#         #yMat_pca   = pca.transform(yMat, nocenter=False)
#         C          = pca.X_center
#         D          = pca.X_scale
#         A          = pca.A[:,0:NVarsRedTemp].T
#         L          = pca.L
#         AT         = A.T
#         yMat_pca   = ((yMat - C)/D).dot(AT)
#         #yMat_      = pca.reconstruct(yMat_pca, nocenter=False)
#         yMat_      = (yMat_pca.dot(A))*D + C
        
#         ySource_pca   = (ySource/D).dot(AT) 
#         ySource_      = (ySource_pca.dot(A))*D 
        
#         RMSE_Mat[i,j]        = np.sqrt( np.mean( (yMat[:,iy] - yMat_[:,iy])**2 ) )
#         RMSE_Source_Mat[i,j] = np.sqrt( np.mean( (ySource[:,iy] - ySource_[:,iy])**2 ) )
#         #RMSE_Mat[i,j]        = np.max(abs(yMat - yMat_))
#         #RMSE_Source_Mat[i,j] = np.max(abs(ySource - ySource_))
        
#         j+=1        
    
#     plt.plot(NVarsRedVec, RMSE_Mat[i,:], '-o', label=ScalingType)
    
#     i+=1

# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'No of Components, $q$')
# plt.ylabel('RMSE on '+VarsName[iy]+' Prediction')
# FigPath = FigDir+'/PCA_'+VarOI+'_Errors.png'
# fig.savefig(FigPath, dpi=900)

# fig = plt.figure(figsize=(16,12))
# i=0
# for ScalingType in ScalingTypeVec:
#     plt.plot(NVarsRedVec, RMSE_Source_Mat[i,:], '-o', label=ScalingType)
#     i+=1
# plt.yscale('log')
# plt.legend()
# plt.xlabel(r'No of Components, $q$')
# plt.ylabel('RMSE on '+VarsName[iy]+' Source Prediction')
# FigPath = FigDir+'/PCA_'+VarOI+'_Source_Errors.png'
# fig.savefig(FigPath, dpi=900)


#['['level', 'pareto', 'range', 'std', 'vast']', 'pareto', 'range', 'std', 'vast']

### 
pca        = PCAA(yMat, scaling='pareto', n_components=NVarsRed)
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



Header   = 'PC_1'
for iVarsRed in range(1,NVarsRed):
    Header += ','+'PC_'+str(iVarsRed+1)
HeaderS  = 'SPC_1'
for iVarsRed in range(1,NVarsRed):
    HeaderS += ','+'SPC_'+str(iVarsRed+1)

#yMat_pca    = pca.transform(yMat, nocenter=False)
yMat_pca   = ((yMat - C)/D).dot(AT)
FileName    = OutputDir+'/'+str(NVarsRed)+'PC/'+str(TrainFlg)+'/ext/PC.csv'
np.savetxt(FileName, np.concatenate((tOrig[...,np.newaxis], yMat_pca), axis=1), delimiter=',', header='t,'+Header, comments='')

#ySource_pca = pca.transform(ySource, nocenter=False)
ySource_pca = (ySource/D).dot(AT) 
FileName    = OutputDir+'/'+str(NVarsRed)+'PC/'+str(TrainFlg)+'/ext/PCSource.csv'
np.savetxt(FileName, np.concatenate((tOrig[...,np.newaxis], ySource_pca), axis=1), delimiter=',', header='t,'+HeaderS, comments='')

FileName    = OutputDir+'/'+str(NVarsRed)+'PC/'+str(TrainFlg)+'/ext/PCAll.csv'
Temp        = np.concatenate((tOrig[...,np.newaxis], yMat_pca, ySource_pca), axis=1)
np.savetxt(FileName, Temp, delimiter=',', header='t,'+Header+','+HeaderS, comments='')

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

    FileName    = OutputDir+'/Orig/'+TrainFlg+'/ext/y.csv.'+str(iRest) 
    Datay       = pd.read_csv(FileName, header=0)
    tVec        = Datay['t'].to_numpy()[...,np.newaxis]
    yTemp       = Datay[KeptSpeciesNames].to_numpy()
    #print('yTemp = ', yTemp)


    yMat_pca    = ((yTemp - C)/D).dot(AT)
    FileName    = OutputDir+'/'+str(NVarsRed)+'PC/'+str(TrainFlg)+'/ext/PC.csv.'+str(iRest)
    Temp        = np.concatenate((tVec, yMat_pca), axis=1)
    np.savetxt(FileName, Temp, delimiter=',', header=Header, comments='')

    FileName    = OutputDir+'/Orig/'+TrainFlg+'/ext/ySource.csv.'+str(iRest) 
    Datay       = pd.read_csv(FileName, header=0)
    tVec        = Datay['t'].to_numpy()[...,np.newaxis]
    ySourceTemp = Datay[KeptSpeciesNames].to_numpy()

    ySource_pca = (ySourceTemp/D).dot(AT) 
    FileName    = OutputDir+'/'+str(NVarsRed)+'PC/'+str(TrainFlg)+'/ext/PCSource.csv.'+str(iRest)
    Temp        = np.concatenate((tVec, ySource_pca), axis=1)
    np.savetxt(FileName, Temp, delimiter=',', header=HeaderS, comments='')
