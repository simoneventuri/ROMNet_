import time
import os
import shutil
import sys
import tensorflow                             as tf
import numpy                                  as np
import pandas                                 as pd

from .model        import Model
from ..training    import customCallbacks, LossHistory
from ..nn          import fnn_block_tfp, deeponet_final_layer_tfp

from tensorflow                           import keras
from tensorflow.keras                     import backend as K
from tensorflow                           import train
from tensorflow.keras                     import layers
from tensorflow.keras                     import optimizers
from tensorflow.keras                     import losses
from tensorflow.keras                     import callbacks

import tensorflow_probability                 as tfp



#=======================================================================================================================================
class DeepONet_BayesByBackprop(Model):     


    # Class Initialization
    def __init__(self, InputData, PathToRunFld, TrainData, ValidData):


        #-------------------------------------------------------------------------------------------------------------------------------  
        self.NVarsBranch = len(InputData.BranchVars)
        self.NVarsTrunk  = len(InputData.TrunkVars)
        self.NVarsx      = self.NVarsBranch + self.NVarsTrunk
        self.NVarsy      = len(InputData.OutputVars)

        if (InputData.TrainIntFlg >= 1):
            self.xTrain       = TrainData[0]
            self.yTrain       = TrainData[1]
            self.xValid       = ValidData[0]
            self.yValid       = ValidData[1]
        else:
            self.xTrain       = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=InputData.BranchVars+InputData.TrunkVars)
            self.yTrain       = pd.DataFrame(np.zeros((1,self.NVarsy)), columns=InputData.OutputVars)
            self.xValid       = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=InputData.BranchVars+InputData.TrunkVars)
            self.yValid       = pd.DataFrame(np.zeros((1,self.NVarsy)), columns=InputData.OutputVars)
        #-------------------------------------------------------------------------------------------------------------------------------  



        if (InputData.DefineModelIntFlg > 0):
            print('[ROMNet]:   Defining ML Model from Scratch')


            if (InputData.TransferFlg): 
                ModelFile         = InputData.TransferModelFld + '/MyModel/'
                NN_Transfer_Model = keras.models.load_model(ModelFile)
                MCFile            = InputData.TransferModelFld + '/Params/ModelCheckpoint/cp-{epoch:04d}.ckpt'
                checkpoint_dir    = os.path.dirname(MCFile)
                latest            = train.latest_checkpoint(checkpoint_dir)
                NN_Transfer_Model.load_weights(latest)
            else:
                NN_Transfer_Model = None



            #---------------------------------------------------------------------------------------------------------------------------
            input_                  = tf.keras.Input(shape=[self.NVarsx,])
            inputBranch, inputTrunk = tf.split(input_, num_or_size_splits=[self.NVarsBranch, self.NVarsTrunk], axis=1)


            NOutputsNN  = self.NVarsy
            if (InputData.SigmaLike is not None):
                CalibrateSigmaLFlg = False
                NOutputsNN         = np.int32(self.NVarsy)
                OutputVars         = InputData.OutputVars

                outputTrunk        = []
                outputTrunk.append(fnn_block_tfp(InputData,  inputTrunk, self.xTrain[InputData.TrunkVars], 'Trunk', 'Trunk_'+InputData.TrunkVars[0],  0, NN_Transfer_Model))

            else:
                CalibrateSigmaLFlg = True
                NOutputsNN         = np.int32(self.NVarsy * 2)
                OutputVars         = InputData.OutputVars + [InputData.OutputVars[i]+'_Sigma' for i in range(self.NVarsy)]

                outputTrunk        = []
                outputTrunk.append(fnn_block_tfp(InputData,  inputTrunk, self.xTrain[InputData.TrunkVars], 'Trunk', 'Trunk_'+str(1),           0, NN_Transfer_Model))
                outputTrunk.append(fnn_block_tfp(InputData,  inputTrunk, self.xTrain[InputData.TrunkVars], 'Trunk', 'Trunk_'+str(1)+'_Sigma',  0, NN_Transfer_Model))


            outputLayer = []
            for iy in range(NOutputsNN):
                jy = np.mod(iy, self.NVarsy)
                ky = iy//self.NVarsy

                ### Branches
                outputBranch       = fnn_block_tfp(InputData, inputBranch, self.xTrain[InputData.BranchVars], 'Branch', 'Branch_'+OutputVars[iy], jy, NN_Transfer_Model)
                
                ## Final Dot Product
                output_P           = layers.Dot(axes=1)([outputBranch, outputTrunk[ky]])
       
                output_net         = deeponet_final_layer_tfp(InputData, output_P, iy, OutputVars, NN_Transfer_Model)

                outputLayer.append(output_net)


            ### If Multiple NNs (i.e., One per Output Variable), then Concatenate their Outputs
            if (self.NVarsy > 1):
                output_final = tf.keras.layers.Concatenate(axis=1)(outputLayer)
            else:
                output_final = outputLayer[0]


            if (CalibrateSigmaLFlg):

                def normal_sp(params): 
                    #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                    dist = tfp.distributions.MultivariateNormalDiag(loc=params[:,0:self.NVarsy], scale_diag=1e-8 + tf.math.softplus(0.05 * params[:,self.NVarsy:])) 
                    return dist
            else:

                def normal_sp(params): 
                    #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                    dist = tfp.distributions.MultivariateNormalDiag(loc=params[:,0:self.NVarsy], scale_diag=InputData.SigmaLike)
                    return dist

            dist       = tfp.layers.DistributionLambda(normal_sp)(output_final) 

            self.Model = keras.Model(inputs=[input_], outputs=[dist] )
            #---------------------------------------------------------------------------------------------------------------------------


            #---------------------------------------------------------------------------------------------------------------------------           
            LearningRate = optimizers.schedules.ExponentialDecay(InputData.LearningRate, decay_steps=InputData.DecaySteps, decay_rate=InputData.DecayRate, staircase=True)
            #LearningRate = InputData.LearningRate

            MTD = InputData.Optimizer
            if (MTD == 'adadelta'):  # A SGD method based on adaptive learning rate
                opt = optimizers.Adadelta(learning_rate=LearningRate, rho=0.95, epsilon=InputData.epsilon, name='Adadelta')
            elif (MTD == 'adagrad'):
                opt = optimizers.Adagrad(learning_rate=LearningRate, initial_accumulator_value=0.1, epsilon=InputData.epsilon, name="Adagrad")
            elif (MTD == 'adam'):    # A SGD method based on adaptive estimation of first-order and second-order moments
                opt = optimizers.Adam(learning_rate=LearningRate, beta_1=InputData.OptimizerParams[0], beta_2=InputData.OptimizerParams[1], epsilon=InputData.OptimizerParams[2], amsgrad=False, name='Adam')
            elif (MTD == 'adamax'):  # Variant of Adam algorithm based on the infinity norm.
                opt = optimizers.Adam(learning_rate=LearningRate, beta_1=InputData.OptimizerParams[0], beta_2=InputData.OptimizerParams[1], epsilon=InputData.OptimizerParams[2], name='Adamax')
            elif (MTD == 'ftrl'):
                opt = optimizers.Ftrl(learning_rate=LearningRate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=kW1, l2_regularization_strength=kW2, name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
            elif (MTD == 'nadam'):   # Variant of Adam algorithm with Nesterov momentum
                opt = optimizers.Nadam(learning_rate=LearningRate, beta_1=InputData.OptimizerParams[0], beta_2=InputData.OptimizerParams[1], epsilon=InputData.OptimizerParams[2], name='Nadam')
            elif (MTD == 'rmsprop'):
                opt = optimizers.RMSprop(learning_rate=LearningRate, rho=0.9, momentum=InputData.OptimizerParams[0], epsilon=InputData.OptimizerParams[1], centered=False, name='RMSprop')
            elif (MTD == 'sgd'):
                opt = optimizers.SGD(learning_rate=LearningRate, momentum=InputData.OptimizerParams[0], nesterov=NestFlg, name="SGD")
            #---------------------------------------------------------------------------------------------------------------------------


            #---------------------------------------------------------------------------------------------------------------------------
            def NLL(y, distr): 
                return K.sum( -distr.log_prob(y) )
            
            LF = InputData.LossFunction
            if (LF == 'NLL'):
                lss = NLL
            #---------------------------------------------------------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Compiling ML Model with Loss and Optimizer')
            self.Model.compile(loss=lss, optimizer=opt)


            ModelFile = PathToRunFld + '/NNModel'
            print('[ROMNet]:   Saving ML Model in File: ' + ModelFile)
            # try:
            #     os.makedirs(ModelFile)
            #     print("\n[ROMNet]: Creating Run Folder ...")
            # except OSError as e:
            #     pass
            #self.Model.save(ModelFile)
            #---------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Summarizing ML Model Structure:')
            self.Model.summary()
            #-------------------------------------------------------------------------------------------------------------------------------

        else:
            
            #---------------------------------------------------------------------------------------------------------------------------
            ModelFile      = PathToRunFld + '/NNModel'
            print('[ROMNet]:   Loading ML Model from Folder: ' + ModelFile)
            self.Model     = keras.models.load_model(ModelFile)
            #---------------------------------------------------------------------------------------------------------------------------

    #===================================================================================================================================



    #===================================================================================================================================
    def train(self, InputData):

        ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
        MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
        LRCallBack    = customCallbacks.customReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, mode='auto', min_delta=1.e-4, cooldown=0, min_lr=1.e-8, verbose=1)
        TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=0, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        
        #CallBacksList = [ESCallBack, MCCallBack, LRCallBack, TBCallBack]
        CallBacksList = [ESCallBack, MCCallBack, TBCallBack]

        History       = self.Model.fit(self.xTrain, self.yTrain, 
                                       batch_size=InputData.MiniBatchSize, 
                                       validation_data=(self.xValid, self.yValid), 
                                       verbose=1, 
                                       epochs=InputData.NEpoch, 
                                       callbacks=CallBacksList)

        loss_history = LossHistory(History.history['loss'], History.history['val_loss'])

        return loss_history
        
    #===================================================================================================================================



    #===================================================================================================================================
    def load_params(self, InputData):

        MCFile         = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(MCFile)
        latest         = train.latest_checkpoint(checkpoint_dir)

        print('[ROMNet]:   Loading ML Model Parameters from File: ', latest)

        self.Model.load_weights(latest)

    #===================================================================================================================================



    #===================================================================================================================================
    def predict(self, xData):

        yPred = self.Model.predict(xData)

        return yPred
    #===================================================================================================================================