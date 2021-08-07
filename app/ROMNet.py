import os
import sys
import tensorflow                             as tf
import numpy                                  as np

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt

if __name__ == "__main__": 

    WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
    ROMNetFldr     = WORKSPACE_PATH + '/ROMNet/romnet/'


    print("\n======================================================================================================================================")
    print(" TensorFlow version: {}".format(tf.__version__))
    print(" Eager execution: {}".format(tf.executing_eagerly()))



    #===================================================================================================================================
    print("\n[ROMNet]: Loading Modules and Functions ...")

    sys.path.insert(0, ROMNetFldr  + '/src/Reading/')
    # from Reading import read_data, read_losseshistory
    sys.path.insert(0, ROMNetFldr  + '/src/Plotting/')
    from Plotting import plot_losseshistory
    # sys.path.insert(0, ROMNetFldr  + '/src/Saving/')
    # from Saving import save_parameters, save_data

    if (len(sys.argv) > 1):
        InputFile = sys.argv[1]
        print("[ROMNet]:   Calling ROMNet with Input File = ", InputFile)
        sys.path.insert(0, InputFile)
    else:
        InputFile = ROMNetFldr + '/src/InputData/'
        print("[ROMNet]:   Calling ROMNet with the PRESET Input File Located in " + InputFile )
        sys.path.insert(0, InputFile)

    #===================================================================================================================================



    #===================================================================================================================================
    print("\n[ROMNet]: Keep Loading Modules and Functions...")
    from ROMNet_Input import inputdata

    print("\n[ROMNet]: Initializing Input ...")
    InputData    = inputdata(WORKSPACE_PATH, ROMNetFldr)

    PathToFldr = InputData.PathToRunFld
    try:
        os.makedirs(PathToFldr)
        print("\n[ROMNet]: Creating Run Folder ...")
    except OSError as e:
        pass
        
    PathToFldr = InputData.PathToFigFld
    try:
        os.makedirs(PathToFldr)
    except OSError as e:
        pass

    #===================================================================================================================================


    #===================================================================================================================================
    print("\n[ROMNet]: Loading Final Modules ... ")

    sys.path.insert(0, ROMNetFldr  + '/src/Model/' + InputData.ApproxModel + '/')
    from Model import model
    # if (InputData.ApproxModel == 'FNN'):
    #     from Model import FNN

    sys.path.insert(0, ROMNetFldr  + '/src/Data/' + InputData.DataType + '/')
    from Data import generate_data

    #===================================================================================================================================



    #===================================================================================================================================
    print("\n[ROMNet]: Generating Data ... ")

    InputData, TrainData, ValidData, AllData, TestData, ExtraData = generate_data(InputData)

    #===================================================================================================================================



    #===================================================================================================================================
    print('\n[ROMNet]: Initializing ML Model ... ')

    NN = model(InputData, InputData.PathToRunFld, TrainData, ValidData)

    #===================================================================================================================================



    #===================================================================================================================================
    if (InputData.TrainIntFlg > 0):


        if (InputData.TrainIntFlg == 1):
            print('\n[ROMNet]: Reading the ML Model Parameters ... ')

            NN.load_params(InputData.PathToParamsFld)


        print('\n[ROMNet]: Training the ML Model ... ')

        History = NN.train(InputData)


        print('\n[ROMNet]: Plotting the Losses Evolution ... ')

        plot_losseshistory(InputData, History)


    else:

        print('\n[ROMNet]: Reading the ML Model Parameters ... ')

        NN.load_params(InputData.PathToParamsFld)

    #===================================================================================================================================



    #===================================================================================================================================

    if (InputData.PlotIntFlg >= 1):

        print('\n[ROMNet]: Evaluating the ML Model at the Training Data and Plotting the Results ... ')

        # xAll      = AllData[0]
        # yAll      = AllData[1]
        # yPred     = NN.Model.predict(xAll.to_numpy())
        
        # iy=0
        # for OutputVar in InputData.OutputVars:
        #     print(OutputVar)

        #     fig = plt.figure(figsize=(16, 12))
        #     plt.plot(xAll['t'], yAll[OutputVar], 'ko')
        #     plt.plot(xAll['t'], yPred[:,iy], 'ro')
        #     plt.xlabel('t')
        #     plt.ylabel(OutputVar)
        #     #plt.xscale('log')
        #     plt.show()
        #     iy+=1


        xAll      = TestData[0]
        yAll      = TestData[1]
        yPred     = NN.Model.predict(xAll.to_numpy())
        
        iy=0
        for OutputVar in InputData.OutputVars:
            print(OutputVar)

            fig = plt.figure(figsize=(16, 12))
            if (InputData.TrunkScale == np.log10):
                plt.plot(10.**xAll['t'], yAll[OutputVar], 'ko')
                plt.plot(10.**xAll['t'], yPred[:,iy], 'ro')
            else:
                plt.plot(xAll['t'], yAll[OutputVar], 'ko')
                plt.plot(xAll['t'], yPred[:,iy], 'ro')
            plt.xlabel('t')
            plt.ylabel(OutputVar)
            #plt.xscale('log')
            plt.show()
            iy+=1

    #===================================================================================================================================



    # # #===================================================================================================================================

    # if (InputData.TestIntFlg >= 1):

    #     if (InputData.PlotIntFlg >= 1):

    #          print('\n[ROMNet]: Evaluating the ML Model at the Test Data and Plotting the Results ... ')

    #          xTest     = TestData[0]
    #          yTest     = TestData[1]
    #          yPred     = NN.Model.predict(xTest[NN.xTrainingVar])

    #          plot_prediction(InputData, 'Test', InputData.TTranVecTest, xTest, yTest, yPred)

    # # #===================================================================================================================================



    # #===================================================================================================================================

    # if (InputData.TestIntFlg >= 1):

    #     if (InputData.PlotIntFlg >= 1):

    #         print('\n[ROMNet]: Evaluating the ML Model at the Test Data and Plotting the Results ... ')

    #         xExtra    = ExtraData
    #         yPred     = NN.Model.predict(xExtra[NN.xTrainingVar])
    #         yData     = []

    #         plot_prediction(InputData, 'Extra', InputData.TTranVecExtra, xExtra, yData, yPred)

    # #===================================================================================================================================


    # #===================================================================================================================================

    # #if (InputData.PredictIntFlg >= 1):
   
    #  #   print('\n[ROMNet]: Generating Rate Matrixes ... ')

    #   #  TTran = 10000.0
    #    # KExcitMat = generate_predictiondata(InputData, NN, TTran)
    #    # print(KExcitMat)
    #    # print(KExcitMat[9,:])

    # #===================================================================================================================================
