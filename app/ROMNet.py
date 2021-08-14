import os
import sys
import tensorflow                             as tf
import numpy                                  as np
from pathlib import Path
import shutil

print(tf.version.VERSION)

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib                           import pyplot as plt


#=======================================================================================================================================
from datetime import datetime

def get_curr_time():
    return datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

def get_start_time():
    return get_curr_time()

#=======================================================================================================================================


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
    InputData              = inputdata(WORKSPACE_PATH, ROMNetFldr)


    print("\n[ROMNet]: Creating Run Folder ...")
    InputData.PathToRunFld = InputData.PathToRunFld + '/' + InputData.SurrogateType + '/' + InputData.ProbApproach + '/'
    path = Path(InputData.PathToRunFld)
    path.mkdir(parents=True, exist_ok=True)

    Prefix = 'Run_'
    if (InputData.NNRunIdx == 0):
        if (len([x for x in os.listdir(InputData.PathToRunFld) if 'Run_' in x]) > 0):
            InputData.NNRunIdx = str(np.amax( np.array( [int(x[len(Prefix):]) for x in os.listdir(InputData.PathToRunFld) if Prefix in x], dtype=int) ) + 1)
        else:
            InputData.NNRunIdx = 1

    InputData.TBCheckpointFldr = InputData.PathToRunFld + '/TB/' + Prefix + str(InputData.NNRunIdx) + "_{}".format(get_start_time())
    print("\n[ROMNet]: TensorBoard Data can be Found here: " + InputData.TBCheckpointFldr)

    InputData.PathToRunFld     = InputData.PathToRunFld +    '/' + Prefix + str(InputData.NNRunIdx)
    path = Path(InputData.PathToRunFld)
    path.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(InputFile+'/ROMNet_Input.py', InputData.PathToRunFld + '/ROMNet_Input.py')
    except OSError as e:
        pass
    
    InputData.PathToFigFld = InputData.PathToRunFld+'/Figures/'
    path = Path(InputData.PathToFigFld)
    path.mkdir(parents=True, exist_ok=True)
    print("\n[ROMNet]: Final Figures can be Found here: " + InputData.PathToFigFld)

    InputData.PathToParamsFld = InputData.PathToRunFld+'/Params/'

    #===================================================================================================================================



    #===================================================================================================================================
    print("\n[ROMNet]: Loading Final Modules ... ")

    SurrogateType = InputData.SurrogateType
    if (SurrogateType == 'FNN-SourceTerms'):
        SurrogateType = 'FNN'

    sys.path.insert(0, ROMNetFldr  + '/src/Model/' + SurrogateType + '/' + InputData.ProbApproach + '/')
    from Model import model

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

            NN.load_params(InputData)


        print('\n[ROMNet]: Training the ML Model ... ')

        History = NN.train(InputData)


        if (InputData.PlotIntFlg >= 1):
    
            print('\n[ROMNet]: Plotting the Losses Evolution ... ')

            plot_losseshistory(InputData, History)


    else:

        print('\n[ROMNet]: Reading the ML Model Parameters ... ')

        NN.load_params(InputData.PathToParamsFld)

    #===================================================================================================================================



    # #===================================================================================================================================

    # if (InputData.PlotIntFlg >= 1):

    #     print('\n[ROMNet]: Evaluating the ML Model at the Training Data and Plotting the Results ... ')

    #     # xAll      = AllData[0]
    #     # yAll      = AllData[1]
    #     # yPred     = NN.Model.predict(xAll.to_numpy())
        
    #     # iy=0
    #     # for OutputVar in InputData.OutputVars:
    #     #     print(OutputVar)

    #     #     fig = plt.figure(figsize=(16, 12))
    #     #     plt.plot(xAll['t'], yAll[OutputVar], 'ko')
    #     #     plt.plot(xAll['t'], yPred[:,iy], 'ro')
    #     #     plt.xlabel('t')
    #     #     plt.ylabel(OutputVar)
    #     #     #plt.xscale('log')
    #     #     plt.show()
    #     #     iy+=1

    #     xAll      = TestData[0]
    #     yAll      = TestData[1]
    #     yPred     = NN.Model.predict(xAll.to_numpy())
    
    #     try:        
    #         iy=0
    #         for OutputVar in InputData.OutputVars:
    #             print(OutputVar)

    #             fig = plt.figure(figsize=(16, 12))
    #             Flg = False
    #             if (InputData.SurrogateType == 'DeepONet'):
    #                 if (InputData.TrunkScale == np.log10):
    #                     Flg = True
    #             if (Flg):
    #                 plt.plot(10.**xAll['t'], yAll[OutputVar], 'ko')
    #                 plt.plot(10.**xAll['t'], yPred[:,iy], 'ro')
    #             else:
    #                 plt.plot(xAll['t'], yAll[OutputVar], 'ko')
    #                 plt.plot(xAll['t'], yPred[:,iy], 'ro')
    #             plt.xlabel('t')
    #             plt.ylabel(OutputVar)
    #             #plt.xscale('log')
    #             plt.show()
    #             iy+=1
    #     except:
    #         pass
    # #===================================================================================================================================



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
