import time
import os
import shutil
import sys
import tensorflow                             as tf
import numpy                                  as np
import pandas                                 as pd 

from tensorflow                           import keras
from tensorflow                           import train
from tensorflow.keras                     import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras                     import regularizers
from tensorflow.keras                     import optimizers
from tensorflow.keras                     import losses
from tensorflow.keras                     import callbacks
from tensorflow.python.ops                import array_ops
from sklearn.model_selection              import train_test_split



#=======================================================================================================================================
@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
class L1Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l1, x0):
    self.l1 = l1
    self.x0 = x0

  def __call__(self, x):
    return self.l1 * tf.math.reduce_sum(tf.math.abs(x - self.x0))

  def get_config(self):
    return {'l1': float(self.l1)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
class L2Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l2, x0):
    self.l2 = l2
    self.x0 = x0

  def __call__(self, x):
    return self.l2 * tf.math.reduce_sum(tf.math.square(x - self.x0))

  def get_config(self):
    return {'l2': float(self.l2)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1l2')
class L1L2Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l1, l2, x0):
    self.l1 = l1
    self.l2 = l2
    self.x0 = x0

  def __call__(self, x):
    Diff = x - self.x0
    return self.l1 * tf.math.reduce_sum(tf.math.abs(Diff)) + self.l2 * tf.math.reduce_sum(tf.math.square(Diff))

  def get_config(self):
    return {'l1l2': float(self.l2)}
#=======================================================================================================================================



#=======================================================================================================================================
def NNBranch(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = InputData.BranchLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.BranchActFun[Idx]

    hiddenVec = [normalized]
    for iLayer in range(NLayers):
        WeightsName = NNName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        if (InputData.TransferFlg):
            x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
            b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
            WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
            bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            WRegul = L1L2Regularizer(kW1, kW2, x0)
            bRegul = L1L2Regularizer(kW1, kW2, b0)
        else:
            WIni   = 'glorot_normal'
            bIni   = 'zeros'
            WRegul = regularizers.l1_l2(l1=kW1, l2=kW2)
            bRegul = regularizers.l1_l2(l1=kW1, l2=kW2)

        hiddenVec.append(layers.Dense(units              = NNLayers[iLayer],
                                      activation         = ActFun[iLayer],
                                      use_bias           = True,
                                      kernel_initializer = WIni,
                                      bias_initializer   = bIni,
                                      kernel_regularizer = WRegul,
                                      bias_regularizer   = bRegul,
                                      name               = LayerName)(hiddenVec[-1]))
        
        if (iLayer < NLayers-1):
            hiddenVec.append( layers.Dropout(InputData.BranchDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.BranchDropOutPredFlg) )          

        if (InputData.BranchSoftmaxFlg):
            hiddenVec.append(layers.Softmax()(hiddenVec[-1]))   

    return hiddenVec[-1]

#=======================================================================================================================================





#=======================================================================================================================================
def NNTrunk(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = InputData.TrunkLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.TrunkActFun[Idx]

    hiddenVec = [normalized]
    for iLayer in range(NLayers):
        WeightsName = NNName + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        if (InputData.TransferFlg):
            x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
            b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
            WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
            bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            WRegul = L1L2Regularizer(kW1, kW2, x0)
            bRegul = L1L2Regularizer(kW1, kW2, b0)
        else:
            WIni   = 'glorot_normal'
            bIni   = 'zeros'
            WRegul = regularizers.l1_l2(l1=kW1, l2=kW2)
            bRegul = regularizers.l1_l2(l1=kW1, l2=kW2)

        hiddenVec.append(layers.Dense(units              = NNLayers[iLayer],
                                      activation         = ActFun[iLayer],
                                      use_bias           = True,
                                      kernel_initializer = WIni,
                                      bias_initializer   = bIni,
                                      kernel_regularizer = WRegul,
                                      bias_regularizer   = bRegul,
                                      name               = LayerName)(hiddenVec[-1]))
        
        if (iLayer < NLayers-1):
            hiddenVec.append( layers.Dropout(InputData.TrunkDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.TrunkDropOutPredFlg) )          

    return hiddenVec[-1]

#=======================================================================================================================================



#=======================================================================================================================================
class BiasLayer(layers.Layer):
    def __init__(self, b0, LayerName):
        super(BiasLayer, self).__init__(name=LayerName)
        self.b0   = b0

    def build(self, input_shape):
        bIni      = tf.keras.initializers.constant(value=self.b0)
        self.bias = self.add_weight('bias',
                                    shape       = input_shape[1:],
                                    initializer = bIni,
                                    trainable   = True)
    def call(self, x):
        return x + self.bias

#=======================================================================================================================================



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

            print('InputData.BranchVars=', InputData.BranchVars)

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


            ### Trunks
            outputTrunk        = NNTrunk(InputData,  InputTrunk, 'Trunk_'+str(1), 0, NN_Transfer_Model)

            outputLayer = []
            for iy in range(self.NVarsy):

                ### Branches
                outputBranch       = NNBranch(InputData, InputBranch, 'Branch_'+InputData.OutputVars[iy], iy, NN_Transfer_Model)
                
                ## Final Dot Product
                output_P           = layers.Dot(axes=1)([outputBranch, outputTrunk])
           
                ### Final Layer
                LayerName      = 'FinalScaling_' + InputData.OutputVars[iy]
                if (InputData.FinalLayerFlg):
                    if (InputData.TransferFlg):
                        x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
                        b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
                        WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                        bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                    else:
                        WIni   = 'glorot_normal'
                        bIni   = 'zeros'
                    output_net = layers.Dense(units              = 1,
                                              activation         = 'linear',
                                              use_bias           = True,
                                              kernel_initializer = WIni,
                                              bias_initializer   = bIni,
                                              name               = LayerName)(output_P) 
                else:
                    b0 = 0
                    if (InputData.TransferFlg): 
                        b0 = NN_Transfer_Model.get_layer(LayerName).bias.numpy()[0]
                    output_net = BiasLayer(b0=b0, LayerName=LayerName)(output_P)

                outputLayer.append(output_net)


            if (self.NVarsy > 1):
                output_final = tf.keras.layers.Concatenate(axis=1)(outputLayer)
            else:
                output_final = outputLayer[0]


            self.Model = keras.Model(inputs=[input_], outputs=[output_final] )
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
            LF = InputData.LossFunction
            if (LF == 'binary_crossentropy'):
                lss = losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy")
            elif (LF == 'categorical_crossentropy'):
                lss = losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction="auto", name="categorical_crossentropy",)
            elif (LF == 'sparse_categorical_crossentropy'):
                lss = losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")
            elif (LF == 'poisson'):
                lss = losses.Poisson(reduction="auto", name="poisson")
            elif (LF == 'binary_crossenkl_divergencetropy'):
                lss = losses.KLDivergence(reduction="auto", name="kl_divergence")
            elif (LF == 'mean_squared_error'):
                lss = losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
            elif (LF == 'mean_absolute_error'):
                lss = losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error")
            elif (LF == 'mean_absolute_percentage_error'):
                lss = losses.MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")
            elif (LF == 'mean_squared_logarithmic_error'):
                lss = losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")
            elif (LF == 'cosine_similarity'):
                lss = losses.CosineSimilarity(axis=-1, reduction="auto", name="cosine_similarity")
            elif (LF == 'huber_loss'):
                lss = losses.Huber(delta=1.0, reduction="auto", name="huber_loss")
            elif (LF == 'log_cosh'):
                lss = losses.LogCosh(reduction="auto", name="log_cosh")
            elif (LF == 'hinge'):
                lss = losses.Hinge(reduction="auto", name="hinge")
            elif (LF == 'squared_hinge'):
                lss = losses.SquaredHinge(reduction="auto", name="squared_hinge")
            elif (LF == 'categorical_hinge'):
                lss = losses.CategoricalHinge(reduction="auto", name="categorical_hinge")
            elif (LF == 'rmse'):
                lss = rmse
            elif (LF == 'rmseexp'):
                lss = rmseexp
            elif (LF == 'rmsenorm'):
                lss = rmsenorm
            #---------------------------------------------------------------------------------------------------------------------------

            #---------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Compiling ML Model with Loss and Optimizer')
            self.Model.compile(loss=lss, optimizer=opt)


            if (InputData.TrainIntFlg >= 1):
                ModelFile = PathToRunFld + '/NNModel'
                print('[ROMNet]:   Saving ML Model in File: ' + ModelFile)
                # try:
                #     os.makedirs(ModelFile)
                #     print("\n[ROMNet]: Creating Run Folder ...")
                # except OSError as e:
                #     pass
                self.Model.save(ModelFile)
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