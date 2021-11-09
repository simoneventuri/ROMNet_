import sys
from os import path
import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split

import matplotlib
#import tkinter
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt


class Data:

    def __init__(self, InputData):

        self.DataType = InputData.DataType

        if (self.DataType == 'BlackBox'):

            self.PathToTrainDataFld = InputData.PathToTrainDataFld
            self.InputFile          = InputData.InputFile
            self.OutputFile         = InputData.OutputFile
            self.dOutputFile        = InputData.dOutputFile

            self.InputVars          = InputData.InputVars
            self.OutputVars         = InputData.OutputVars

            self.SurrogateType      = InputData.SurrogateType
            if (self.SurrogateType == 'DeepONet'):
                self.BranchVars         = InputData.BranchVars
                self.BranchScale        = InputData.BranchScale
                self.TrunkVars          = InputData.TrunkVars
                self.TrunkScale         = InputData.TrunkScale

            self.TestPerc           = InputData.TestPerc
            self.ValidPerc          = InputData.ValidPerc

            self.PINN               = InputData.PINN

            self.get                = self.read



    #=======================================================================================================================================
    # Reading Data 
    def read(self):
        print('[SurQCT]:   Reading Data from: ' + self.PathToTrainDataFld)

        Data = pd.read_csv(self.PathToTrainDataFld+'/'+self.InputFile, header=0)


        if (self.SurrogateType == 'DeepONet'):

            uAll = Data[self.BranchVars]
            if (len(self.TrunkVars) > 0):
                tAll         = Data[self.TrunkVars]
                xAll         = pd.concat([uAll, tAll], axis=1)
            else:
                xAll         = uAll

            if (not self.BranchScale == None):
                for Var in self.BranchVars:
                    xAll[Var] = xAll[Var].apply(lambda x: self.BranchScale(x+1.e-15))
            if (not self.TrunkScale == None):
                for Var in self.TrunkVars:
                    xAll[Var] = xAll[Var].apply(lambda x: self.TrunkScale(x+1.e-15))

        else:
            xAll = Data[self.InputVars]


       

        for iCol in range(xAll.shape[1]):
            array_sum = np.sum(xAll.to_numpy()[:,iCol])
            if (np.isnan(array_sum)):
                print('xAll has NaN!!!')

        Data = pd.read_csv(self.PathToTrainDataFld+'/'+self.OutputFile, header=0)
        yAll = Data[self.OutputVars]
        for iCol in range(yAll.shape[1]):
            array_sum = np.sum(yAll.to_numpy()[:,iCol])
            if (np.isnan(array_sum)):
                print('yAll has NaN!!!')


        if (self.PINN):
            Data  = pd.read_csv(self.PathToTrainDataFld+'/'+self.dOutputFile, header=0)
            dyAll = Data[self.OutputVars]
            dOutputVars = []
            for iy in range(len(self.OutputVars)):
                dOutputVars.append('d'+self.OutputVars[iy])
            dyAll.columns = dOutputVars
            for iCol in range(dyAll.shape[1]):
                array_sum = np.sum(dyAll.to_numpy()[:,iCol])
                if (np.isnan(array_sum)):
                    print('dyAll has NaN!!!')
            yAll  = pd.concat([yAll, dyAll], axis=1)

        print('xAll = ', xAll.head())
        print('yAll = ', yAll.head())


        xTrain     = xAll.copy()
        xTest      = xAll.copy()
        xTrain     = xTrain.sample(frac=(1.0-self.TestPerc/100.0), random_state=3)
        xTest      = xTest.drop(xTrain.index)
        xValid     = xTrain.copy()
        xTrain     = xTrain.sample(frac=(1.0-self.ValidPerc/100.0), random_state=3)
        xValid     = xValid.drop(xTrain.index)

        yTrain     = yAll.copy()
        yTest      = yAll.copy()
        yTrain     = yTrain.sample(frac=(1.0-self.TestPerc/100.0), random_state=3)
        yTest      = yTest.drop(yTrain.index)
        yValid     = yTrain.copy()
        yTrain     = yTrain.sample(frac=(1.0-self.ValidPerc/100.0), random_state=3)
        yValid     = yValid.drop(yTrain.index)

        self.Train = [xTrain, yTrain]
        self.Valid = [xValid, yValid]
        self.All   = [xAll,   yAll]
        self.Test  = [xTest,  yTest]
        self.Extra = []


        # for OutputVar in self.OutputVars:

        #     fig = plt.figure(figsize=(16, 12))
        #     plt.plot(xTrain['t'], yTrain[OutputVar], 'o')
        #     plt.plot(xValid['t'], yValid[OutputVar], 'o')
        #     plt.xlabel('t')
        #     plt.ylabel(OutputVar)
        #     #plt.xscale('log')
        #     plt.show()


    #=======================================================================================================================================

