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
def NN(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = InputData.Layers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.ActFun[Idx]

    hiddenVec = [normalized]
    for iLayer in range(NLayers):
        WeightsName = NNName + '_' + InputData.OutputVars[Idx] + '_HL' + str(iLayer+1) 
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
        
        # if (iLayer < NLayers-1):
        #     hiddenVec.append( layers.Dropout(InputData.DropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.DropOutPredFlg) )          

    return hiddenVec[-1]

#=======================================================================================================================================


#=======================================================================================================================================
class model:    


    # Class Initialization
    def __init__(self, InputData, PathToRunFld, TrainData, ValidData):


        #-------------------------------------------------------------------------------------------------------------------------------  
        # Extract weights and biases
        def ExtractParmaters(NN):

            weights    = []
            biases     = []
            activation = []
            for layer in NN.layers:
                weightBias = layer.get_weights()
                
                if (len(weightBias) == 2):
                    weights.append(weightBias[0].T)
                    
                    bias = weightBias[1]
                    bias = np.reshape(bias, (len(bias),1))
                    biases.append(bias)
                
                # if (len(weightBias) == 1):
                #     activation.append(weightBias[0])

            return weights, biases
        #-------------------------------------------------------------------------------------------------------------------------------  



        #-------------------------------------------------------------------------------------------------------------------------------  
        self.NVarsx      = len(InputData.InputVars)
        self.NVarsy      = len(InputData.OutputVars)

        if (InputData.TrainIntFlg >= 1):
            self.xTrain       = TrainData[0]
            self.yTrain       = TrainData[1]
            self.xValid       = ValidData[0]
            self.yValid       = ValidData[1]
        else:
            self.xTrain       = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=InputData.InputVars)
            self.yTrain       = pd.DataFrame(np.zeros((1,self.NVarsy)), columns=InputData.OutputVars)
            self.xValid       = pd.DataFrame(np.zeros((1,self.NVarsx)), columns=InputData.InputVars)
            self.yValid       = pd.DataFrame(np.zeros((1,self.NVarsy)), columns=InputData.OutputVars)
        #-------------------------------------------------------------------------------------------------------------------------------  


        if (InputData.DefineModelIntFlg > 0):
            print('[ROMNet]:   Defining Deterministic Model from Scratch')


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
            
            ### Input Layer
            input_              = tf.keras.Input(shape=[self.NVarsx,])


            ### Normalizer Layer
            if (InputData.NormalizeInput):
                if (InputData.TransferFlg): 
                    Mean        = NN_Transfer_Model.get_layer('normalization').mean.numpy()
                    Variance    = NN_Transfer_Model.get_layer('normalization').variance.numpy()
                    normalizer  = preprocessing.Normalization(mean=Mean, variance=Variance)
                else:
                    normalizer  = preprocessing.Normalization()
                    normalizer.adapt(np.array(self.xTrain))
                Input_          = normalizer(input_)
            else:
                Input_          = input_


            NNNs        = len(InputData.Layers)
            NOutputsNN  = self.NVarsy
            if (NNNs > 1):
                NOutputsNN = 1
            outputLayer = []
            for iy in range(NNNs):

                ### Hidden Layers
                output_L       = NN(InputData, Input_, 'NN', iy, NN_Transfer_Model)
                       
                ### Final Layer
                LayerName      = 'FinalScaling_' + str(iy+1)
                if (InputData.TransferFlg):
                    x0     = NN_Transfer_Model.get_layer(LayerName).kernel.numpy()
                    b0     = NN_Transfer_Model.get_layer(LayerName).bias.numpy()
                    WIni   = tf.keras.initializers.RandomNormal(mean=x0, stddev=1.e-10)
                    bIni   = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
                else:
                    WIni   = 'glorot_normal'
                    bIni   = 'zeros'
                output_net = layers.Dense(units              = NOutputsNN,
                                          activation         = 'linear',
                                          use_bias           = True,
                                          kernel_initializer = WIni,
                                          bias_initializer   = bIni,
                                          name               = LayerName)(output_L) 
                
                outputLayer.append(output_net)


            ### If Multiple NNs (i.e., One per Output Variable), then Concatenate their Outputs
            if (NNNs > 1):
                output_final = tf.keras.layers.Concatenate(axis=1)(outputLayer)
            else:
                output_final = outputLayer[0]


            self.PreModel = keras.Model(inputs=[input_], outputs=[output_final] )
            #---------------------------------------------------------------------------------------------------------------------------


            #---------------------------------------------------------------------------------------------------------------------------           
            sys.path.append(InputData.ROMNetFldr + '/src/Callbacks/')
            from customCallbacks import customReduceLROnPlateau

            ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
            MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
            MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
            LRCallBack    = customReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, mode='auto', min_delta=1.e-4, cooldown=0, min_lr=1.e-8, verbose=1)
            TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=0, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
            
            #CallBacksList = [ESCallBack, MCCallBack, LRCallBack, TBCallBack]
            CallBacksList = [ESCallBack]


            if (InputData.TrainIntFlg >= 1):
                NCycles = InputData.NCycles
            else:
                NCycles = 1 

            for iCycle in range(NCycles):

                #---------------------------------------------------------------------------------------------------------------------------           
                #LearningRate = optimizers.schedules.ExponentialDecay(InputData.LearningRate, decay_steps=InputData.DecaySteps, decay_rate=InputData.DecayRate, staircase=True)
                LearningRate = InputData.LearningRate * (10**(-iCycle)) 

                MTD = InputData.Optimizer
                if (MTD == 'adadelta'):  # A SGD method based on adaptive learning rate
                    opt = optimizers.Adadelta(learning_rate=LearningRate, rho=0.95, epsilon=InputData.epsilon, name='Adadelta')
                elif (MTD == 'adagrad'):
                    opt = optimizers.Adagrad(learning_rate=LearningRate, initial_accumulator_value=0.1, epsilon=InputData.epsilon, name="Adagrad")
                elif (MTD == 'adam'):    # A SGD method based on adaptive estimation of first-order and second-order moments
                    opt = optimizers.Adam(learning_rate=LearningRate, beta_1=InputData.OptimizerParams[0], beta_2=InputData.OptimizerParams[1], epsilon=InputData.OptimizerParams[2], amsgrad=True, name='Adam')
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
                print('[ROMNet]:   Compiling DeterministicModel with Loss and Optimizer')
                self.PreModel.compile(loss=lss, optimizer=opt)
                #---------------------------------------------------------------------------------------------------------------------------

                #-------------------------------------------------------------------------------------------------------------------------------
                print('[ROMNet]:   Summarizing Deterministic Model Structure:')
                self.PreModel.summary()
                #-------------------------------------------------------------------------------------------------------------------------------

                if (InputData.TrainIntFlg >= 1):
    
                    #-------------------------------------------------------------------------------------------------------------------------------
                    print('[ROMNet]:   Pretraining Deterministic Model:')
                    History       = self.PreModel.fit(self.xTrain, self.yTrain, 
                                                      batch_size=InputData.MiniBatchSize, 
                                                      validation_data=(self.xValid, self.yValid), 
                                                      verbose=1, 
                                                      epochs=InputData.NEpoch, 
                                                      callbacks=CallBacksList)
                    #-------------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            if (InputData.TrainIntFlg >= 1):
                
                ModelFile = PathToRunFld + '/NNPreModel'
                print('[ROMNet]:   Saving Deterministic Model in File: ' + ModelFile)
                # try:
                #     os.makedirs(ModelFile)
                #     print("\n[ROMNet]: Creating Run Folder ...")
                # except OSError as e:
                #     pass
                self.PreModel.save(ModelFile)
            #-------------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Extracting Parameters from the Pretrained Model:')

            weights, biases = ExtractParmaters(self.PreModel)
            #-------------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Importing TensorBNN Library:')

            sys.path.append(InputData.ROMNetFldr+'/extra/TensorBNN/')
            #sys.path.append('/Users/sventuri/WORKSPACE/uqml/uq4nn/external/TensorBNN/')
            from tensorBNN.activationFunctions import Tanh, Relu
            from tensorBNN.layer               import DenseLayer                  as DenseLayer_TBNN
            from tensorBNN.network             import network                     as network_TBNN
            from tensorBNN.likelihood          import GaussianLikelihood          as GaussianLikelihood_TBNN
            from tensorBNN.metrics             import SquaredError                as SquaredError_TBNN
            from tensorBNN.metrics             import PercentError                as PercentError_TBNN
            
            dtype_TBNN = tf.float32
            Seed_TBNN  = 1000
            #-------------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Constructing the BNN:')

            self.Model  = network_TBNN(dtype_TBNN, self.NVarsx, self.xTrain.to_numpy(), self.yTrain.to_numpy()[:,0:self.NVarsy], self.xValid.to_numpy(), self.yValid.to_numpy()[:,0:self.NVarsy], InputData.PathToRunFld) 

            NNLayers    = InputData.Layers[0]
            NLayers     = len(NNLayers)
            ActFun      = InputData.ActFun[0]
            NNeuronsTot = np.concatenate((np.array([self.NVarsx]), NNLayers, np.array([self.NVarsy])))

            Seed = Seed_TBNN
            for iLayer in range(NLayers):
                WeightsName = 'BNN_HL' + str(iLayer+1) 
                LayerName   = WeightsName 
            
                self.Model.add( DenseLayer_TBNN(NNeuronsTot[iLayer], NNeuronsTot[iLayer+1], seed=Seed, dtype=dtype_TBNN, weights=weights[iLayer], biases=biases[iLayer], activation='linear', name='dense') )
                
                if (ActFun[iLayer] == 'tanh'):
                    self.Model.add(Tanh())
                elif (ActFun[iLayer] == 'relu'):
                    self.Model.add(Relu())
                elif (ActFun[iLayer] == 'sigmoid'):
                    self.Model.add(Sigmoid())
                
                Seed += 1000

            LayerName = 'FinalScaling_'+str(1)
            self.Model.add( DenseLayer_TBNN(NNeuronsTot[-2], NNeuronsTot[-1], seed=Seed, dtype=dtype_TBNN, weights=weights[-1], biases=biases[-1], activation='linear', name='dense') )

            self.Likelihood = GaussianLikelihood_TBNN(sd=InputData.SigmaLike)
            #-------------------------------------------------------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------------------------------------------------------
            print('[ROMNet]:   Setting Up HMC:')

            self.Model.setupMCMC(InputData.StpStart,        
                                 InputData.StpMin,         
                                 InputData.StpMax,         
                                 InputData.StpAdapt,        
                                 InputData.LeapStart,        
                                 InputData.LeapMin,         
                                 InputData.LeapMax,         
                                 InputData.LeapBtwn,        
                                 InputData.StpHyper,        
                                 InputData.LeapHyper,        
                                 InputData.NEpochsBurnIn,        
                                 InputData.NCores,         
                                 InputData.NAveraging)
            
            NormInfo        = (0,1) # mean, sd
            self.MetricList = [SquaredError_TBNN(mean=NormInfo[0], sd=NormInfo[1]), PercentError_TBNN(mean=NormInfo[0], sd=NormInfo[1])]
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

        self.FolderName = InputData.PathToRunFld + '/NNModel'
        self.Model.train(InputData.NEpochsHMC,                    # epochs to train for
                         InputData.SaveStep,                      # increment between network saves
                         self.Likelihood,
                         metricList      = self.MetricList,
                         folderName      = self.FolderName,       # Name of folder for saved networks
                         networksPerFile = InputData.NNPerFile)   # Number of networks saved per file

        History = None

        return History

    #===================================================================================================================================



    #===================================================================================================================================
    def load_params(self, InputData):

        sys.path.append(InputData.ROMNetFldr+'/extra/TensorBNN/')
        from tensorBNN.predictor import predictor as predictor_TBNN

        latest = InputData.PathToRunFld+'/NNModel/'

        print('[ROMNet]:   Loading ML Model Parameters from File: ', latest)
        self.Model = predictor_TBNN(latest, tf.float32)

    #===================================================================================================================================



    #===================================================================================================================================
    def predict(self, xData):

        yPred = self.Model.predict(xData, n=0)[0]

        return yPred
    #===================================================================================================================================


    #===================================================================================================================================
    def predic_samples(self, xData, NSamples):

        yPredSampled = np.concatenate( self.Model.predict(xData, n=1), axis=1)
        
        return yPredSampled[:,self.NVarsy*NSamples]
    #===================================================================================================================================