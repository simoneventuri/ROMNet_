import os
import sys
import numpy  as np
import shutil

import romnet as rmnt



if __name__ == "__main__": 

    WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
    ROMNetFldr     = WORKSPACE_PATH + '/ROMNet/romnet/'

    

    #===========================================================================
    print("\n[ROMNet]: Loading Input Module ...")

    try:
        InputFile = sys.argv[1]
        print("[ROMNet]:   Calling ROMNet with Input File = ", InputFile)
        sys.path.insert(0, InputFile)
    except OSError:
        print('Input File not Specified')


    from ROMNet_Input import inputdata

    print("\n[ROMNet]: Initializing Input ...")
    InputData               = inputdata(WORKSPACE_PATH, ROMNetFldr)
    InputData.InputFilePath = InputFile
    #===========================================================================



    #===========================================================================
    print("\n[ROMNet]: Importing Physical System ... ")

    if (InputData.PhysSystem is not None):
        System = getattr(rmnt.pinn.system, InputData.PhysSystem)
        system = System(InputData)
    #===========================================================================



    #===========================================================================
    print("\n[ROMNet]: Getting Data ... ")

    Data = getattr(rmnt.data, InputData.DataType)
    data = Data(InputData, system)
    data.get(InputData)

    #===========================================================================



    #===========================================================================
    SurrogateType = InputData.SurrogateType
    if (SurrogateType == 'FNN-SourceTerms'):
        SurrogateType = 'FNN'
        
    Net   = getattr(rmnt.nn, SurrogateType)

    model = rmnt.model.Model_Deterministic(InputData)

    model.build(InputData, data, Net)

    #===========================================================================



    #===========================================================================
    if (InputData.TrainIntFlg > 0):

        model.compile(InputData)

        model.train(InputData)


        if (InputData.PlotIntFlg >= 1):
    
            print('\n[ROMNet]: Plotting the Losses Evolution ... ')

            #loss_history.plot(InputData)


    else:

        print('\n[ROMNet]: Reading the ML Model Parameters ... ')

        model.load_params(InputData.PathToParamsFld)

    #===========================================================================