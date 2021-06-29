import sys
import os, errno
import os.path
from os import path

import numpy as np
import tensorflow as tf
import h5py


##### Saving Parameters
def save_parameters(model, InputData):

    PathToFldr = InputData.PathToParamsFld
    try:
        os.makedirs(PathToFldr)
    except OSError as e:
        pass

    #print('[ProPDE]:   Saving Final Biases')
    biasFinalBest = model.biasFinalBest
    PathToFile = PathToFldr + '/FinalBiases.csv'
    save_biases(PathToFile, tf.squeeze(biasFinalBest))


    NBranches = InputData.BranchLayers.shape[0]
    for iBranch in range(NBranches):
        weights    = model.weightsBranchBest[iBranch]
        biases     = model.biasesBranchBest[iBranch]

        NetName = 'Branch' + str(iBranch+1)
<<<<<<< HEAD
        #print('[ProPDE]:   Saving Parameters for ' + NetName)
=======
        print('[ProPDE]:   Saving Parameters for Net ' + NetName)
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
        PathToFldr = InputData.PathToParamsFld + NetName + '/'
        try:
            os.makedirs(PathToFldr)
        except OSError as e:
            pass

        #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
        ActScaling = model.ActScalingBranchBest[iBranch]
        PathToFile = PathToFldr + 'ActScaling.csv'
        save_actscaling(PathToFile, ActScaling)

        NLayers = InputData.BranchLayers.shape[1]
        for iLayer in range(NLayers-1):
<<<<<<< HEAD
            PathToFldr = InputData.PathToParamsFld + NetName + '/HL' + str(iLayer+1) + '/'
=======
            if iLayer == 0:
                LayerName = 'InputLayer'
            else:
                LayerName = 'HiddenLayer' + str(iLayer)

            PathToFldr = InputData.PathToParamsFld + NetName + '/' + LayerName + '/'
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
            try:
                os.makedirs(PathToFldr)
            except OSError as e:
                pass

            if ( (iLayer == NLayers-2) and not(InputData.BiasBranchFlg) ):
<<<<<<< HEAD
                #print('[ProPDE]:   Saving Weights for Layer ' + str(iLayer+1))
=======
                print('[ProPDE]:     Saving Weights for ' + LayerName)
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
                PathToFile = PathToFldr + 'Weights.npz'
                np.savez(PathToFile,     weights[iLayer])
                PathToFile = PathToFldr + 'Weights.csv'
                save_weigths(PathToFile, weights[iLayer])
            else:
<<<<<<< HEAD
                #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
=======
                print('[ProPDE]:     Saving Weights and Biases for ' + LayerName)
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
                PathToFile = PathToFldr + 'Weights.npz'
                np.savez(PathToFile,     weights[iLayer], biases[iLayer])
                PathToFile = PathToFldr + 'Weights.csv'
                save_weigths(PathToFile, weights[iLayer])
                PathToFile = PathToFldr + 'Biases.csv'
                save_biases(PathToFile,  biases[iLayer])
            
            

    NetName = 'Trunk'
<<<<<<< HEAD
    #print('[ProPDE]:   Saving Parameters for ' + NetName)
=======
    print('[ProPDE]:   Saving Parameters for Net ' + NetName)
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
    PathToFldr = InputData.PathToParamsFld + NetName + '/'
    try:
        os.makedirs(PathToFldr)
    except OSError as e:
        pass
    weights    = model.weightsTrunkBest
    biases     = model.biasesTrunkBest

    #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
    ActScaling = model.ActScalingTrunkBest
    PathToFile = PathToFldr + 'ActScaling.csv'
    save_actscaling(PathToFile, ActScaling)

    NLayers = InputData.TrunkLayers.shape[0]
    for iLayer in range(NLayers-1):
<<<<<<< HEAD
        PathToFldr = InputData.PathToParamsFld + NetName + '/HL' + str(iLayer+1) + '/'
=======
        if iLayer == 0:
            LayerName = 'InputLayer'
        else:
            LayerName = 'HiddenLayer' + str(iLayer)

        PathToFldr = InputData.PathToParamsFld + NetName + '/' + LayerName + '/'
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
        try:
            os.makedirs(PathToFldr)
        except OSError as e:
            pass

<<<<<<< HEAD
        #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
=======
        print('[ProPDE]:     Saving Weights and Biases for ' + LayerName)
>>>>>>> c94cae8a81afa8855c29b2edc1871e5922d9f7e4
        PathToFile = PathToFldr + 'Weights.npz'
        np.savez(PathToFile,     weights[iLayer], biases[iLayer])
        PathToFile = PathToFldr + 'Weights.csv'
        save_weigths(PathToFile, weights[iLayer])
        PathToFile = PathToFldr + 'Biases.csv'
        save_biases(PathToFile,  biases[iLayer])


##### Saving Parameters
def save_parameters_hdf5(model, InputData):

    PathToFldr = InputData.PathToParamsFld
    try:
        os.makedirs(PathToFldr)
    except OSError as e:
        pass
    PathToFile    = PathToFldr + '/Params.hdf5'
    HDF5Exist_Flg = path.exists(PathToFile)
    #if (HDF5Exist_Flg):
    f = h5py.File(PathToFile, 'a')
    #else:
    #    f = {'key': 'value'}


    biasFinalBest = model.biasFinalBest
    CheckStr      = 'Finalb'
    if CheckStr in f.keys():
        Data          = f['Finalb']
        Data[...]     = biasFinalBest.numpy()
    else:
        f.create_dataset("Finalb", data=biasFinalBest.numpy(), compression="gzip", compression_opts=9)


    NBranches = InputData.BranchLayers.shape[0]
    for iBranch in range(NBranches):
        #print('[ProPDE]:   Saving Parameters for Branch ' + str(iBranch))
        CheckStr = '/Branch' + str(iBranch+1) + '/'

        ActScaling = model.ActScalingBranchBest[iBranch]
        weights    = model.weightsBranchBest[iBranch]
        biases     = model.biasesBranchBest[iBranch]

        if CheckStr in f.keys():
            grp       = f[CheckStr]

            #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
            Data      = grp["ActScaling"]
            Data[...] = np.array([ActScaling.numpy()])

            NLayers = InputData.TrunkLayers.shape[0]
            for iLayer in range(NLayers-1):
                #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
                CheckStrTemp  = CheckStr + '/HL' + str(iLayer+1) + '/'
                Data          = grp[CheckStrTemp+'W']
                Data[...]     = weights[iLayer].numpy()
                Data          = grp[CheckStrTemp+'b']
                Data[...]     = biases[iLayer].numpy()
                
        else:
            grp           = f.create_group(CheckStr)

            #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
            ActScaling_ = grp.create_dataset("ActScaling", data=np.array([ActScaling.numpy()]), compression="gzip", compression_opts=9)

            NLayers = InputData.TrunkLayers.shape[0]
            for iLayer in range(NLayers-1):
                #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
                CheckStrTemp = CheckStr + '/HL' + str(iLayer+1) + '/'
                grp      = f.create_group(CheckStrTemp)
                W_       = grp.create_dataset("W", data=weights[iLayer].numpy(), compression="gzip", compression_opts=9)
                b_       = grp.create_dataset("b", data=biases[iLayer].numpy(),  compression="gzip", compression_opts=9)


    CheckStr = '/Trunk/'
    if CheckStr in f.keys():
        #print('[ProPDE]:   Saving Parameters for Trunk')

        grp       = f[CheckStr]

        #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
        Data      = grp["ActScaling"]
        Data[...] = np.array([model.ActScalingTrunkBest[0].numpy()])

        NLayers = InputData.TrunkLayers.shape[0]
        for iLayer in range(NLayers-1):
            #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
            CheckStrTemp  = CheckStr + '/HL' + str(iLayer+1) + '/'
            Data          = grp[CheckStrTemp+'W']
            Data[...]     = model.weightsTrunkBest[iLayer].numpy()
            Data          = grp[CheckStrTemp+'b']
            Data[...]     = model.biasesTrunkBest[iLayer].numpy()
            
    else:
        grp           = f.create_group(CheckStr)

        #print('[ProPDE]:   Saving Activation Functions Scaling Parameters')
        ActScaling_ = grp.create_dataset("ActScaling", data=np.array([model.ActScalingTrunkBest[0].numpy()]), compression="gzip", compression_opts=9)

        NLayers = InputData.TrunkLayers.shape[0]
        for iLayer in range(NLayers-1):
            #print('[ProPDE]:   Saving Weights and Biases for Layer ' + str(iLayer+1))
            CheckStrTemp = CheckStr + '/HL' + str(iLayer+1) + '/'
            grp      = f.create_group(CheckStrTemp)
            W_       = grp.create_dataset("W", data=model.weightsTrunkBest[iLayer].numpy(), compression="gzip", compression_opts=9)
            b_       = grp.create_dataset("b", data=model.biasesTrunkBest[iLayer].numpy(),  compression="gzip", compression_opts=9)


    f.close()    



def save_weigthsandbiases(PathToWeightsFldr, WAll, bAll):

    PathToFinalW = PathToWeightsFldr + '/W.csv'
    np.savetxt(PathToFinalW, WAll, delimiter=",")

    PathToFinalb = PathToWeightsFldr + '/b.csv'
    np.savetxt(PathToFinalb, np.transpose(bAll), delimiter=",")



def save_weigths(PathToFinalW, WAll):
    np.savetxt(PathToFinalW, WAll.numpy(), delimiter=",")



def save_biases(PathToFinalb, bAll):
    np.savetxt(PathToFinalb, np.transpose(bAll.numpy()), delimiter=",")



def save_actscaling(PathToActScaling, ActScaling):
    np.savetxt(PathToActScaling, np.array([ActScaling]))



def save_data(PathToData, Data):
    np.savetxt(PathToData, Data, delimiter=",")
