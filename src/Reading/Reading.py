import sys
from os import path
import pandas as pd
import numpy  as np


#=======================================================================================================================================
# Reading Data 
def read_data(DataFile, xVarsVec, Suffix):
    print('[SurQCT]:   Reading Molecular Levels Data from: ' + DataFile)



    DataFldr = '/Users/sventuri/WORKSPACE/PCA/Test_DeepONet/deeponet/'

    Data = pd.read_csv(DataFldr+'/Input.csv',header=0)
    tAll = Data.to_numpy()[:,0]
    uAll = Data.to_numpy()[:,1:]

    Data = pd.read_csv(DataFldr+'/Output.csv',header=0)
    xAll = Data.to_numpy()

    return xMat

#=======================================================================================================================================

