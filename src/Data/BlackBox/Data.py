import sys
from os import path
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

import matplotlib
import tkinter
import matplotlib
matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt


#=======================================================================================================================================
# Reading Data 
def generate_data(InputData):
    print('[SurQCT]:   Reading Data from: ' + InputData.PathToDataFld)

    Data = pd.read_csv(InputData.PathToDataFld+'/Input.csv',header=0)

    uAll = Data[InputData.BranchVars]
    if (len(InputData.TrunkVars) > 0):
        tAll         = Data[InputData.TrunkVars]
        xAll         = pd.concat([uAll, tAll], axis=1)
    else:
        xAll         = uAll

    if (not InputData.BranchScale == None):
        for Var in InputData.BranchVars:
            xAll[Var] = xAll[Var].apply(lambda x: InputData.BranchScale(x+1.e-15))
    if (not InputData.TrunkScale == None):
        for Var in InputData.TrunkVars:
            xAll[Var] = xAll[Var].apply(lambda x: InputData.TrunkScale(x+1.e-15))


    for iCol in range(xAll.shape[1]):
        array_sum = np.sum(xAll.to_numpy()[:,iCol])
        if (np.isnan(array_sum)):
            print('xAll has NaN!!!')

    Data = pd.read_csv(InputData.PathToDataFld+'/Output.csv',header=0)
    yAll = Data[InputData.OutputVars]
    for iCol in range(yAll.shape[1]):
        array_sum = np.sum(yAll.to_numpy()[:,iCol])
        if (np.isnan(array_sum)):
            print('yAll has NaN!!!')


    if (InputData.PINN):
        Data  = pd.read_csv(InputData.PathToDataFld+'/dOutput.csv',header=0)
        dyAll = Data[InputData.OutputVars]
        dOutputVars = []
        for iy in range(len(InputData.OutputVars)):
            dOutputVars.append('d'+InputData.OutputVars[iy])
        dyAll.columns = dOutputVars
        for iCol in range(dyAll.shape[1]):
            array_sum = np.sum(dyAll.to_numpy()[:,iCol])
            if (np.isnan(array_sum)):
                print('dyAll has NaN!!!')
        yAll  = pd.concat([yAll, dyAll], axis=1)

    print('xAll = ', xAll.head())
    print('yAll = ', yAll.head())


    xTrain    = xAll.copy()
    xTest     = xAll.copy()
    xTrain    = xTrain.sample(frac=(1.0-InputData.TestPerc/100.0), random_state=3)
    xTest     = xTest.drop(xTrain.index)
    xValid    = xTrain.copy()
    xTrain    = xTrain.sample(frac=(1.0-InputData.ValidPerc/100.0), random_state=3)
    xValid    = xValid.drop(xTrain.index)

    yTrain    = yAll.copy()
    yTest     = yAll.copy()
    yTrain    = yTrain.sample(frac=(1.0-InputData.TestPerc/100.0), random_state=3)
    yTest     = yTest.drop(yTrain.index)
    yValid    = yTrain.copy()
    yTrain    = yTrain.sample(frac=(1.0-InputData.ValidPerc/100.0), random_state=3)
    yValid    = yValid.drop(yTrain.index)

    TrainData = [xTrain, yTrain]
    ValidData = [xValid, yValid]
    AllData   = [xAll,   yAll]
    TestData  = [xTest,  yTest]
    ExtraData = []


    # for OutputVar in InputData.OutputVars:

    #     fig = plt.figure(figsize=(16, 12))
    #     plt.plot(xTrain['t'], yTrain[OutputVar], 'o')
    #     plt.plot(xValid['t'], yValid[OutputVar], 'o')
    #     plt.xlabel('t')
    #     plt.ylabel(OutputVar)
    #     #plt.xscale('log')
    #     plt.show()


    return InputData, TrainData, ValidData, AllData, TestData, ExtraData
#=======================================================================================================================================

