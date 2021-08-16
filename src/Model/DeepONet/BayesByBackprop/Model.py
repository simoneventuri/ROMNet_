import time
import os
import shutil
import sys
import tensorflow                             as tf
import numpy                                  as np
import pandas                                 as pd

from tensorflow                           import keras
from tensorflow.keras                     import backend as K
from tensorflow                           import train
from tensorflow.keras                     import layers
from tensorflow.keras.layers              import Layer
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras                     import regularizers
from tensorflow.keras                     import optimizers
from tensorflow.keras                     import losses
from tensorflow.keras                     import callbacks
from tensorflow.keras                     import activations
from tensorflow.keras                     import initializers
from tensorflow.python.ops                import array_ops

import tensorflow_probability                 as tfp


#=======================================================================================================================================
def NNBranch(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
    bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)

    NNLayers = InputData.BranchLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.BranchActFun[Idx]

    hiddenVec = [normalized]
    for iLayer in range(NLayers):
        WeightsName = NNName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        hiddenVec.append(tfp.layers.DenseFlipout(units                = np.int32(NNLayers[iLayer]),
                                                 activation           = ActFun[iLayer],
                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                 bias_divergence_fn   = bias_divergence_fn,
                                                 name                 = LayerName)(hiddenVec[-1]))


        # if (iLayer < NLayers-1):
        #     hiddenVec.append( layers.Dropout(InputData.BranchDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.BranchDropOutPredFlg) )          

        if (InputData.BranchSoftmaxFlg):
            hiddenVec.append(layers.Softmax()(hiddenVec[-1]))   

    return hiddenVec[-1]

#=======================================================================================================================================





#=======================================================================================================================================
def NNTrunk(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
    bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)

    NNLayers = InputData.TrunkLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.TrunkActFun[Idx]

    hiddenVec = [normalized]
    for iLayer in range(NLayers):
        WeightsName = NNName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        hiddenVec.append(tfp.layers.DenseFlipout(units                = np.int32(NNLayers[iLayer]),
                                                 activation           = ActFun[iLayer],
                                                 bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                 bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                 kernel_divergence_fn = kernel_divergence_fn,
                                                 bias_divergence_fn   = bias_divergence_fn,
                                                 name                 = LayerName)(hiddenVec[-1]))
        
        # if (iLayer < NLayers-1):
        #     hiddenVec.append( layers.Dropout(InputData.TrunkDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.TrunkDropOutPredFlg) )          

    return hiddenVec[-1]

#=======================================================================================================================================



# #=======================================================================================================================================
# class BiasLayer(layers.Layer):
#     def __init__(self, b0, LayerName):
#         super(BiasLayer, self).__init__(name=LayerName)
#         self.b0   = b0

#     def build(self, input_shape):
#         bIni      = tf.keras.initializers.constant(value=self.b0)
#         self.bias = self.add_weight('bias',
#                                     shape       = input_shape[1:],
#                                     initializer = bIni,
#                                     trainable   = True)
#     def call(self, x):
#         return x + self.bias

# #=======================================================================================================================================



#=======================================================================================================================================
class model:    


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


            ### Normalizer Layers
            if (InputData.NormalizeInput):
                if (InputData.TransferFlg): 
                    Mean             = NN_Transfer_Model.get_layer('normalization').mean.numpy()
                    Variance         = NN_Transfer_Model.get_layer('normalization').variance.numpy()
                    normalizerBranch = preprocessing.Normalization(mean=Mean, variance=Variance)
                else:
                    normalizerBranch = preprocessing.Normalization()
                    self.xTrainBranch= self.xTrain[InputData.BranchVars]
                    normalizerBranch.adapt(np.array(self.xTrainBranch))
                InputBranch          = normalizerBranch(inputBranch)
                #
                if (InputData.TransferFlg): 
                    Mean             = NN_Transfer_Model.get_layer('normalization_1').mean.numpy()
                    Variance         = NN_Transfer_Model.get_layer('normalization_1').variance.numpy()
                    normalizerTrunk  = preprocessing.Normalization(mean=Mean, variance=Variance)
                else:
                    normalizerTrunk  = preprocessing.Normalization()
                    self.xTrainTrunk = self.xTrain[InputData.TrunkVars]
                    normalizerTrunk.adapt(np.array(self.xTrainTrunk))
                InputTrunk           = normalizerTrunk(inputTrunk)
            else:
                InputBranch          = inputBranch
                InputTrunk           = inputTrunk


            NOutputsNN  = self.NVarsy
            if (InputData.SigmaLike is not None):
                CalibrateSigmaLFlg = False
                NOutputsNN         = np.int32(self.NVarsy)
                OutputVars         = InputData.OutputVars

                outputTrunk        = []
                outputTrunk.append(NNTrunk(InputData,  InputTrunk, 'Trunk_'+InputData.TrunkVars[0],  0, NN_Transfer_Model))

            else:
                CalibrateSigmaLFlg = True
                NOutputsNN         = np.int32(self.NVarsy * 2)
                OutputVars         = InputData.OutputVars + [InputData.OutputVars[i]+'_Sigma' for i in range(self.NVarsy)]

                outputTrunk        = []
                outputTrunk.append(NNTrunk(InputData,  InputTrunk, 'Trunk_'+str(1),           0, NN_Transfer_Model))
                outputTrunk.append(NNTrunk(InputData,  InputTrunk, 'Trunk_'+str(1)+'_Sigma',  0, NN_Transfer_Model))


            outputLayer = []
            for iy in range(NOutputsNN):
                jy = np.mod(iy, self.NVarsy)
                ky = iy//self.NVarsy

                ### Branches
                outputBranch       = NNBranch(InputData, InputBranch, 'Branch_'+OutputVars[iy], jy, NN_Transfer_Model)
                
                ## Final Dot Product
                output_P           = layers.Dot(axes=1)([outputBranch, outputTrunk[ky]])
       
                kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)
                bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (InputData.MiniBatchSize * 1.0)

                ### Final Layer
                LayerName      = 'FinalScaling_' + OutputVars[iy]
                # if (InputData.FinalLayerFlg):
                output_net     = tfp.layers.DenseFlipout(units                = 1,
                                                         activation           = 'linear',
                                                         bias_posterior_fn    = tfp.layers.util.default_mean_field_normal_fn(),
                                                         bias_prior_fn        = tfp.layers.default_multivariate_normal_fn,
                                                         kernel_divergence_fn = kernel_divergence_fn,
                                                         bias_divergence_fn   = bias_divergence_fn,
                                                         name                 = LayerName)(output_P)


                # else:
                #     b0 = 0
                #     if (InputData.TransferFlg): 
                #         b0 = NN_Transfer_Model.get_layer(LayerName).bias.numpy()[0]
                #     output_net = BiasLayer(b0=b0, LayerName=LayerName)(output_P)

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

        sys.path.append(InputData.ROMNetFldr + '/src/Callbacks/')
        from customCallbacks import customReduceLROnPlateau

        ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
        MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
        LRCallBack    = customReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, mode='auto', min_delta=1.e-4, cooldown=0, min_lr=1.e-8, verbose=1)
        TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=0, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        
        #CallBacksList = [ESCallBack, MCCallBack, LRCallBack, TBCallBack]
        CallBacksList = [ESCallBack, MCCallBack, TBCallBack]

        History       = self.Model.fit(self.xTrain, self.yTrain, 
                                       batch_size=InputData.MiniBatchSize, 
                                       validation_data=(self.xValid, self.yValid), 
                                       verbose=1, 
                                       epochs=InputData.NEpoch, 
                                       callbacks=CallBacksList)

        return History

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