import os
import sys
import numpy                                  as np
from pathlib import Path
import shutil

import romnet as rmnt

#import tensorflow                             as tf
#print(tf.version.VERSION)



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


    # print("\n======================================================================================================================================")
    # print(" TensorFlow version: {}".format(tf.__version__))
    # print(" Eager execution: {}".format(tf.executing_eagerly()))



    #===================================================================================================================================
    print("\n[ROMNet]: Loading Input Module ...")

    try:
        InputFile = sys.argv[1]
        print("[ROMNet]:   Calling ROMNet with Input File = ", InputFile)
        sys.path.insert(0, InputFile)
    except OSError:
        print('Input File not Specified')


    from ROMNet_Input import inputdata

    print("\n[ROMNet]: Initializing Input ...")
    InputData              = inputdata(WORKSPACE_PATH, ROMNetFldr)
    #===================================================================================================================================



    #===================================================================================================================================
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
    print("\n[ROMNet]: Getting Data ... ")

    data = rmnt.data.Data(InputData)
    data.get()

    #===================================================================================================================================



    #===================================================================================================================================
    print('\n[ROMNet]: Initializing ML Model ... ')

    SurrogateType = InputData.SurrogateType
    if (SurrogateType == 'FNN-SourceTerms'):
        SurrogateType = 'FNN'
        
    Model = getattr(rmnt.model,  SurrogateType + '_' + InputData.ProbApproach)
    NN    = Model(InputData, InputData.PathToRunFld, data.Train, data.Valid)

    #===================================================================================================================================



    #===================================================================================================================================
    if (InputData.TrainIntFlg > 0):


        if (InputData.TrainIntFlg == 1):
            print('\n[ROMNet]: Reading the ML Model Parameters ... ')

            NN.load_params(InputData)


        print('\n[ROMNet]: Training the ML Model ... ')

        loss_history = NN.train(InputData)


        if (InputData.PlotIntFlg >= 1):
    
            print('\n[ROMNet]: Plotting the Losses Evolution ... ')

            loss_history.plot(InputData)


    else:

        print('\n[ROMNet]: Reading the ML Model Parameters ... ')

        NN.load_params(InputData.PathToParamsFld)

    #===================================================================================================================================