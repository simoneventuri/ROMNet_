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
pd.options.mode.chained_assignment = None  # default='warn'


### Importing Matplotlib and Its Style

import matplotlib.pyplot                 as plt

#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/presentation.mplstyle'))
#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/zoomed.mplstyle'))
plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/paper_1column.mplstyle'))
#plt.style.use(os.path.join(WORKSPACE_PATH, 'ROMNet/romnet/extra/postprocessing/paper_2columns.mplstyle'))


from PCAfold         import PCA          as PCAA

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA

from sklearn.model_selection import train_test_split



########################################################################################################################
DataDir            = os.path.join(WORKSPACE_PATH, 'ROMNet/Data/TransTanh_100Cases/')
FigDir             = None #os.path.join(WORKSPACE_PATH, '../Desktop/Paper_Figures_DeepONet_/')

n_ics              = 100

tStratch           = 1.
SOLVER             = 'BDF'

valid_perc         = 20.

FixedMinVal        = 1.e-32

DRAlog             = 'PCA'

VarName            = ''
if (VarName == 'All'):
    NModesFinal    = 8
    DRType         = 'All'
else:
    NModesFinal    = 8
    DRType         = 'OneByOne'
    
    
ColorVec           = ['#190707', '#dd3232', '#0065a9', '#348a00','#985396','#f68b69']
########################################################################################################################




Vars       = ['x']
NVars      = len(Vars)
print('Vars = ', Vars)



for VarName in Vars:
    iVar = list(Vars).index(VarName)
    print('\n[DR] Var = ', VarName)

    if (DRType == 'All'):
        
        Cols   = []
        NewFlg = True
        for iC in range(n_ics):


            FileName             = DataDir+'/Orig/train/ext/y.csv.'+str(iC+1)
            Data                 = pd.read_csv(FileName, header=0)
            DataTemp             = Data[[Vars[iVar] for iVar in range(NVars)]]
            DataTemp             = np.maximum(DataTemp, FixedMinVal)


            Flg = True
            for iVar in range(NModesFinal):
                if (np.abs( (DataTemp[Vars[iVar]][0] - DataTemp[Vars[iVar]][len(Data)-1])/DataTemp[Vars[iVar]][0] ) < 1.e-6): 
                    Flg = False
                    break
            
            if Flg:
                DataICTemp  = Data[[Vars[iVar] for iVar in range(NVars)]].iloc[0]
                
                if (NewFlg):
                    DataInput        = DataTemp
                    DataIC           = DataICTemp
                    NewFlg           = False
                else:
                    DataInput        = pd.concat([DataInput, DataTemp], axis=1)
                    DataIC           = pd.concat([DataIC, DataICTemp], axis=1)
                Cols += [str(iC+1)+'_'+str(iVar+1) for iVar in range(NVars)]
                
            else:
                print('iC ', iC)

        tVec              = Data['t']
        DataInput.columns = Cols
        
        yMat              = DataInput.to_numpy()
        
    else:
        
        Cols   = []
        NewFlg = True
        for iC in range(n_ics):



            FileName             = DataDir+'/Orig/train/ext/y.csv.'+str(iC+1)
            Data                 = pd.read_csv(FileName, header=0)
            DataTemp             = np.maximum(Data[Vars[iVar]], FixedMinVal)

            if (np.abs( (DataTemp[0] - DataTemp[len(Data)-1])/DataTemp[0] ) > 1.e-4):
                DataICTemp  = Data[[Vars[iVar] for iVar in range(NVars)]].iloc[0]
                
                if (NewFlg):
                    DataInput        = DataTemp
                    DataIC           = DataICTemp
                    NewFlg           = False
                else:
                    DataInput        = pd.concat([DataInput, DataTemp], axis=1)
                    DataIC           = pd.concat([DataIC, DataICTemp], axis=1)
                Cols.append(str(iC+1))

        tVec              = Data['t']
        DataInput.columns = Cols

        yMat              = DataInput.to_numpy()
        
    DataIC            = DataIC.T.reset_index(drop=True, inplace=False)
    Vars0             = [Var+'0' for Var in Vars]
    DataIC.columns    = Vars0
    ICs               = DataIC.to_numpy()



    def Norm(yMat, C, D):
        return ( yMat - C ) / D
        #return ( yMat - C ) 
        
    def NormInv(yMat_, C, D):
        return yMat_ * D + C 
        #return yMatt + C 

    C     = yMat.mean() * np.ones(yMat.shape[1]) #yMat.mean(axis=0)
    D     = 1.0         * np.ones(yMat.shape[1]) #yMat.std(axis=0)
    yMatt = Norm(yMat, C, D)



    if   (DRAlog == 'PCA'):
        NPCA      = NModesFinal
        
        # PCA_      = PCA(n_components=NPCA, )
        # yMat_DR   = PCA_.fit_transform(yMatt)
        # yMat_     = PCA_.inverse_transform(yMat_DR)

        pca       = PCAA(yMat, scaling='none', n_components=int(NPCA), nocenter=True)
        #C         = pca.X_center
        #D         = pca.X_scale
        A         = pca.A[:,0:NPCA].T
        L         = pca.L
        LL        = np.maximum(L,0.)
        AT        = A.T
        yMat_DR   = yMatt.dot(AT)
        yMat_     = yMat_DR.dot(A)

    elif (DRAlog == 'IPCA'):
        NIPCA     = NModesFinal
        IPCA_     = FastICA(n_components=NIPCA)
        yMat_DR   = IPCA_.fit_transform(yMatt)
        yMat_     = IPCA_.inverse_transform(yMat_DR)

    elif (DRAlog == 'KPCA'):
        NKPCA     = NModesFinal
        #KPCA_     = KernelPCA(n_components=NKPCA, kernel='rbf', degree=5, alpha=0.0001, fit_inverse_transform=True)
        KPCA_     = KernelPCA(n_components=NKPCA, kernel='poly', degree=4, alpha=0.001, fit_inverse_transform=True)
        yMat_DR   = KPCA_.fit(yMatt).transform(yMatt)
        yMat_     = KPCA_.inverse_transform(yMat_DR)
    
    yMat_     = NormInv(yMat_, C, D)

    print('[DR]   Shape of yMat_ = ', yMat_DR.shape)
    print('[DR]   Max % Error  = ', np.max(abs((yMat - yMat_)/yMat)*100))
    print('[DR]   Max      SE  = ', np.max((yMat - yMat_)**2))
    print('[DR]   Mean % Error = ', np.mean(abs((yMat - yMat_)/yMat)*100))
    print('[DR]            MSE = ', np.mean((yMat - yMat_)**2))


    Vars_Branch = ['b_'+str(i_mode+1)  for i_mode in range(NModesFinal)] + ['c','d']
    Vars_Trunk  = ['t_'+str(i_mode+1)  for i_mode in range(NModesFinal)]


    Data             = pd.DataFrame(yMat_DR, columns=Vars_Trunk)
    tVec[tVec == 0.] = FixedMinVal
    Data['t']        = tVec
    Data['log(t)']   = np.log(tVec)
    Data['log10(t)'] = np.log10(tVec)


    if   (DRAlog == 'PCA'):

    ####################################################################################################################
    ### Writing Branches     
        
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1))
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/')
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/train/')
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/valid/')
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/test/')
        except:
            pass



        data_id    = 'pts'

        DataNoZero           = DataIC
        n_points             = len(DataNoZero)

        idx                  = np.arange(n_points)
        train_idx, valid_idx = train_test_split(idx, test_size=valid_perc/100, random_state=42)

        n_valid              = len(valid_idx)
        n_train              = len(train_idx)


        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/train/'+data_id+'/')
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/valid/'+data_id+'/')
        except:
            pass
        try:
            os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/test/'+data_id+'/')
        except:
            pass

        print(DataIC.head())

        DataInput  = DataIC[Vars0]
        DataInput.iloc[train_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/train/'+data_id+'/Input.csv', index=False)
        DataInput.iloc[valid_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/valid/'+data_id+'/Input.csv', index=False)
        DataInput.to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/test/'+data_id+'/Input.csv', index=False)

        DataOutput = pd.DataFrame(np.concatenate([A.T, C[...,np.newaxis], D[...,np.newaxis]], axis=1), columns=Vars_Branch)
        DataOutput.iloc[train_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/train/'+data_id+'/Output.csv', index=False)
        DataOutput.iloc[valid_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/valid/'+data_id+'/Output.csv', index=False)
        DataOutput.to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Branch/test/'+data_id+'/Output.csv', index=False)
        
    ####################################################################################################################
    


    ####################################################################################################################
    ### Writing Trunks

    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1))
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/')
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/train/')
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/valid/')
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/test/')
    except:
        pass



    data_id    = 'pts'

    DataNoZero           = Data[Data['t'] >= FixedMinVal]
    n_points             = len(DataNoZero)

    idx                  = np.arange(n_points)
    train_idx, valid_idx = train_test_split(idx, test_size=valid_perc/100, random_state=42)

    n_valid              = len(valid_idx)
    n_train              = len(train_idx)


    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/train/'+data_id+'/')
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/valid/'+data_id+'/')
    except:
        pass
    try:
        os.makedirs(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/test/'+data_id+'/')
    except:
        pass

    DataInput  = DataNoZero[['t', 'log10(t)', 'log(t)'] + Vars_Trunk]
    DataInput.iloc[train_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/train/'+data_id+'/Input.csv', index=False)
    DataInput.iloc[valid_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/valid/'+data_id+'/Input.csv', index=False)
    DataInput.to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/test/'+data_id+'/Input.csv', index=False)

    DataOutput = DataNoZero[['t', 'log10(t)', 'log(t)'] + Vars_Trunk]
    DataOutput.iloc[train_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/train/'+data_id+'/Output.csv', index=False)
    DataOutput.iloc[valid_idx].to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/valid/'+data_id+'/Output.csv', index=False)
    DataOutput.to_csv(DataDir+'/'+str(NModesFinal)+DRAlog+'/'+str(DRType)+'/Var'+str(iVar+1)+'/Trunk/test/'+data_id+'/Output.csv', index=False)
    
    ####################################################################################################################