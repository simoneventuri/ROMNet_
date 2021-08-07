import time
import os
import shutil
import sys
import tensorflow                             as tf
import numpy                                  as np
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

# from Plotting import plot_var
# from Reading  import read_parameters_hdf5
# from Saving   import save_parameters, save_parameters_hdf5, save_data


#=======================================================================================================================================
def NNBranch(InputData, normalized, NNName, Idx):

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = InputData.BranchLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.BranchActFun[Idx]

    hiddenVec = [normalized]

    for iLayer in range(NLayers):
        LayerName = NNName + str(Idx) + '_HL' + str(iLayer+1) 
        hiddenVec.append(layers.Dense(units=NNLayers[iLayer],
                                activation=ActFun[iLayer],
                                use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros',
                                kernel_regularizer=regularizers.l1_l2(l1=kW1, l2=kW2),
                                bias_regularizer=regularizers.l1_l2(l1=kW1, l2=kW2),
                                name=LayerName)(hiddenVec[-1]))
        if (iLayer < NLayers-1):
            hiddenVec.append(layers.Dropout(InputData.BranchDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.BranchDropOutTrain))      

    hiddenVec.append(layers.Softmax()(hiddenVec[-1]))    

    return hiddenVec[-1]

#=======================================================================================================================================



#=======================================================================================================================================
def NNTrunk(InputData, normalized, NNName, Idx):

    kW1      = InputData.WeightDecay[0]
    kW2      = InputData.WeightDecay[1]
    NNLayers = InputData.TrunkLayers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.TrunkActFun[Idx]

    hiddenVec = [normalized]

    for iLayer in range(NLayers):
        LayerName = NNName + str(Idx) + '_HL' + str(iLayer+1) 
        hiddenVec.append(layers.Dense(units=NNLayers[iLayer],
                                activation=ActFun[iLayer],
                                use_bias=True,
                                kernel_initializer='glorot_normal',
                                bias_initializer='zeros',
                                kernel_regularizer=regularizers.l1_l2(l1=kW1, l2=kW2),
                                bias_regularizer=regularizers.l1_l2(l1=kW1, l2=kW2),
                                name=LayerName)(hiddenVec[-1]))
        if (iLayer < NLayers-1):
            hiddenVec.append(layers.Dropout(InputData.TrunkDropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.TrunkDropOutTrain))          

    return hiddenVec[-1]

#=======================================================================================================================================



#=======================================================================================================================================
class model:    


    # Class Initialization
    def __init__(self, InputData, PathToRunFld, TrainData, ValidData):


        #===================================================================================================================================
        class AdditiveGaussNoise(layers.Layer):
          def __init__(self, mean, stddev):
            super(AdditiveGaussNoise, self).__init__()
            self.mean   = mean
            self.stddev = stddev

          def build(self, input_shape):
            self.input_shape_  = input_shape 
            
          def call(self, input):
            return input + tf.random.normal(shape=array_ops.shape(input), mean=self.mean, stddev=self.stddev)

        #===================================================================================================================================


        #-------------------------------------------------------------------------------------------------------------------------------  
        if (InputData.TrainIntFlg >= 1):
            self.xTrain       = TrainData[0]
            self.yTrain       = TrainData[1]
            self.xValid       = ValidData[0]
            self.yValid       = ValidData[1]
        #-------------------------------------------------------------------------------------------------------------------------------  



        if (InputData.DefineModelIntFlg > 0):
            print('[ROMNet]:   Defining ML Model from Scratch')

            self.NVarsBranch = len(InputData.BranchVars)
            self.NVarsTrunk  = len(InputData.TrunkVars)
            self.NVarsx      = self.NVarsBranch + self.NVarsTrunk
            self.NVarsy      = len(InputData.OutputVars)


            #---------------------------------------------------------------------------------------------------------------------------
            input_                  = tf.keras.Input(shape=[self.NVarsx,])
            inputBranch, inputTrunk = tf.split(input_, num_or_size_splits=[self.NVarsBranch, self.NVarsTrunk], axis=1)
            xTrunk_Train            = self.xTrain[InputData.TrunkVars]


            if (InputData.PINN):
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(inputTrunk)

                    # ### Trunks Normalizer
                    # normalizerTrunk    = preprocessing.Normalization()
                    # normalizerTrunk.adapt(np.array(xTrunk_Train))
                    # normalizedTrunk    = normalizerTrunk(inputTrunk)

                    ### Trunks
                    outputTrunk        = NNTrunk(InputData,  inputTrunk,  'Trunk',  0)

                    ### Branches Normalizer
                    xBranch_Train      = self.xTrain[InputData.BranchVars]
                    normalizerBranch   = preprocessing.Normalization()
                    normalizerBranch.adapt(np.array(xBranch_Train))
                    normalizedBranch   = normalizerBranch(inputBranch)

                    outputLayer = []
                    for iy in range(self.NVarsy):

                        ### Branches
                        outputBranch       = NNBranch(InputData, normalizedBranch, 'Branch', iy)
                        
                        ## Final Dot Product
                        output1            = layers.Dot(axes=1)([outputBranch, outputTrunk])
                       
                        ### Final Layer
                        LayerName          = 'FinalScaling' + str(iy)
                        outputLayer.append(layers.Dense(units=1,
                                                          activation='linear',
                                                          use_bias=True,
                                                          kernel_initializer='glorot_normal',
                                                          bias_initializer='zeros',
                                                          name=LayerName)(output1))

                    if (self.NVarsy > 1):
                        outputNN = tf.concat(outputLayer, axis=1)
                    else:
                        outputNN = outputLayer[0]


                    print(outputNN.shape[1])
                    outputNN_Split = tf.split(outputNN, num_or_size_splits=outputNN.shape[1], axis=1)

                print('sdasda', outputNN_Split[0])
                print('adsda', tape.gradient(outputNN_Split[0], inputTrunk) )     
                doutputNN      = tf.concat([tape.gradient(outputNN_i, inputTrunk) for outputNN_i in outputNN_Split], axis=1)
                output_final   = tf.keras.layers.Concatenate(axis=1)([outputNN, doutputNN])


            else:


                ### Trunks Normalizer
                normalizerTrunk    = preprocessing.Normalization()
                normalizerTrunk.adapt(np.array(xTrunk_Train))
                normalizedTrunk    = normalizerTrunk(inputTrunk)

                ### Trunks
                outputTrunk        = NNTrunk(InputData,  normalizedTrunk,  'Trunk',  0)

                ### Branches Normalizer
                xBranch_Train      = self.xTrain[InputData.BranchVars]
                normalizerBranch   = preprocessing.Normalization()
                normalizerBranch.adapt(np.array(xBranch_Train))
                normalizedBranch   = normalizerBranch(inputBranch)

                outputLayer = []
                for iy in range(self.NVarsy):

                    ### Branches
                    outputBranch       = NNBranch(InputData, normalizedBranch, 'Branch', iy)
                    
                    ## Final Dot Product
                    output1            = layers.Dot(axes=1)([outputBranch, outputTrunk])
                   
                    ### Final Layer
                    LayerName          = 'FinalScaling' + str(iy)
                    outputLayer.append(layers.Dense(units=1,
                                                      activation='linear',
                                                      use_bias=True,
                                                      kernel_initializer='glorot_normal',
                                                      bias_initializer='zeros',
                                                      name=LayerName)(output1))

                if (self.NVarsy > 1):
                    output_final = tf.keras.layers.Concatenate(axis=1)(outputLayer)
                else:
                    output_final = outputLayer[0]


            self.Model = keras.Model(inputs=[input_], outputs=[output_final] )
            #---------------------------------------------------------------------------------------------------------------------------


            #---------------------------------------------------------------------------------------------------------------------------
            LearningRate = optimizers.schedules.ExponentialDecay(InputData.LearningRate, decay_steps=InputData.DecaySteps, decay_rate=InputData.DecayRate, staircase=True)

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

        ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
        MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
        #LRCallBack    = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=500, mode='auto', min_delta=1.e-6, cooldown=0, min_lr=1.e-8, verbose=1)
        TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=100, batch_size=InputData.MiniBatchSize, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        CallBacksList = [ESCallBack, TBCallBack, MCCallBack]

        History       = self.Model.fit(self.xTrain, self.yTrain, 
                                       batch_size=InputData.MiniBatchSize, 
                                       validation_data=(self.xValid, self.yValid), 
                                       verbose=1, 
                                       epochs=InputData.NEpoch, 
                                       callbacks=CallBacksList)

        return History

    #===================================================================================================================================



    #===================================================================================================================================
    def load_params(self, PathToParamsFld):

        MCFile         = PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(MCFile)
        latest         = train.latest_checkpoint(checkpoint_dir)

        print('[ROMNet]:   Loading ML Model Parameters from File: ', latest)

        self.Model.load_weights(latest)

    #===================================================================================================================================



