import os
import sys
import numpy  as np
import shutil

import romnet as rmnt



if __name__ == "__main__": 

    try:
        WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
        print("\n[ROMNet.py                          ]: Found WORKSPACE_PATH Environmental Variable: WORKSPACE_PATH = ", WORKSPACE_PATH)
    except:
        WORKSPACE_PATH = None
    

    #===========================================================================
    print("\n[ROMNet.py                          ]: Loading Input Module ...")

    try:
        InputFile = sys.argv[1]
        print("[ROMNet.py                          ]:   Calling ROMNet with Input File = ", InputFile)
        sys.path.insert(0, InputFile)
    except OSError:
        print('Input File not Specified')


    from ROMNet_Input import inputdata

    print("\n[ROMNet.py                          ]: Initializing Input ...")
    InputData               = inputdata(WORKSPACE_PATH)
    InputData.InputFilePath = InputFile
    #===========================================================================



    #===========================================================================
    print("\n[ROMNet.py                          ]: Importing Physical System ... ")

    if (InputData.phys_system is not None):
        System = getattr(rmnt.pinn.system, InputData.phys_system)
        system = System(InputData)
    #===========================================================================

              
              
    #===========================================================================
    print("\n[ROMNet.py                          ]: Getting Data ... ")

    Data = getattr(rmnt.data, InputData.data_type)
    data = Data(InputData, system)
    data.get(InputData)
    #===========================================================================



    #===========================================================================
    surrogate_type = InputData.surrogate_type
    if (surrogate_type == 'FNN-SourceTerms'):
        surrogate_type = 'FNN'
        
    Net   = getattr(rmnt.architecture, surrogate_type)

    model = rmnt.model.Model_TF(InputData)

    model.build(InputData, data, Net, system)

    #===========================================================================



    #===========================================================================
    if (InputData.train_int_flg > 0):

        model.compile(InputData)

        model.train(InputData)


    else:

        print('\n[ROMNet.py                          ]: Reading the ML Model Parameters ... ')

        model.load_params(InputData.PathToParamsFld)

    #===========================================================================