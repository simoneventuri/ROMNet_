import time
import os
import shutil
import sys
import tensorflow                             as tf
import numpy                                  as np

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

from sklearn.model_selection              import train_test_split



#=======================================================================================================================================
def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w, prior_params):
    comp_1_dist   = tfp.distributions.Normal(0.0, prior_params[0])
    comp_2_dist   = tfp.distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    




class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, PriorSigma, PriorPi, activation=None, **kwargs):
        self.output_dim     = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation     = activations.get(activation)
        self.PriorSigma     = PriorSigma
        self.PriorPi        = PriorPi
        super().__init__(**kwargs)

    def build(self, input_shape):  
        
        # Mixture prior parameters shared across DenseVariational layer instances
        self.prior_params, self.prior_sigma = mixture_prior_params(sigma_1=self.PriorSigma[0], sigma_2=self.PriorSigma[1], pi=self.PriorPi[0])

        self._trainable_weights.append(self.prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer=initializers.RandomNormal(stddev=self.prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer=initializers.RandomNormal(stddev=self.prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', 
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel       = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma   = tf.math.softplus(self.bias_rho)
        bias         = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)
                
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w, self.prior_params))
#=======================================================================================================================================



#=======================================================================================================================================
def NN(InputData, normalized, NNName, Idx, NN_Transfer_Model):

    NNLayers = InputData.Layers[Idx]
    NLayers  = len(NNLayers)
    ActFun   = InputData.ActFun[Idx]

    hiddenVec = [normalized]
    #LayEr    = normalized
    for iLayer in range(NLayers):
        WeightsName = NNName + '_' + InputData.OutputVars[Idx] + '_HL' + str(iLayer+1) 
        LayerName   = WeightsName 

        hiddenVec.append(DenseVariational(output_dim         = NNLayers[iLayer],
                                          activation         = ActFun[iLayer],
                                          kl_loss_weight     = InputData.KLLossWeight,
                                          PriorSigma         = InputData.PriorSigma,
                                          PriorPi            = InputData.PriorPi,
                                          name               = LayerName)(hiddenVec[-1]))
        # LayEr = DenseVariational(NNLayers[iLayer],
        #                         activation         = ActFun[iLayer],
        #                         kl_loss_weight     = InputData.KLLossWeight,
        #                         name               = LayerName)(LayEr)

        # if (iLayer < NLayers-1):
        #     hiddenVec.append( layers.Dropout(InputData.DropOutRate, input_shape=(NNLayers[iLayer],))(hiddenVec[-1], training=InputData.DropOutPredFlg) )          

    return hiddenVec[-1]

#=======================================================================================================================================


#=======================================================================================================================================
class model:    


    # Class Initialization
    def __init__(self, InputData, PathToRunFld, TrainData, ValidData):

        
        #-------------------------------------------------------------------------------------------------------------------------------  
        self.NVarsx      = len(InputData.InputVars)
        self.NVarsy      = len(InputData.OutputVars)

        if (InputData.TrainIntFlg >= 1):
            self.xTrain       = TrainData[0]
            self.yTrain       = TrainData[1]
            self.xValid       = ValidData[0]
            self.yValid       = ValidData[1]
        else:
            self.xTrain       = np.array([[1]*self.NVarsx ], dtype=np.float64)
            self.yTrain       = np.array([[1]*self.NVarsy ], dtype=np.float64)
            self.xValid       = np.array([[1]*self.NVarsx ], dtype=np.float64)
            self.yValid       = np.array([[1]*self.NVarsy ], dtype=np.float64)
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
            
            ### Input Layer
            #input_              = tf.keras.layers.Input(shape=(self.NVarsx,))
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
                output_net     = DenseVariational(NOutputsNN,
                                                  activation         = 'linear',
                                                  kl_loss_weight     = InputData.KLLossWeight,
                                                  PriorSigma         = InputData.PriorSigma,
                                                  PriorPi            = InputData.PriorPi,
                                                  name               = LayerName)(output_L)

                outputLayer.append(output_net)


            ### If Multiple NNs (i.e., One per Output Variable), then Concatenate their Outputs
            if (NNNs > 1):
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
            SigmaVec = [InputData.SigmaLike] * self.NVarsy
            def neg_log_likelihood(y_obs, y_pred, sigma=SigmaVec):
                #dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
                dist = tfp.distributions.MultivariateNormalDiag(loc=y_pred, scale_diag=SigmaVec)
                return K.sum(-dist.log_prob(y_obs))
            
            
            LF       = InputData.LossFunction
            if (LF == 'neg_log_likelihood'):
                lss = neg_log_likelihood
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

        sys.path.append(InputData.ROMNetFldr + '/src/Callbacks/')
        from customCallbacks import customReduceLROnPlateau

        ESCallBack    = callbacks.EarlyStopping(monitor='val_loss', min_delta=InputData.ImpThold, patience=InputData.NPatience, restore_best_weights=True, mode='auto', baseline=None, verbose=1)
        MCFile        = InputData.PathToParamsFld + "/ModelCheckpoint/cp-{epoch:04d}.ckpt"
        Monitor = 'val_loss'
        if (InputData.ValidPerc == 0.):
            Monitor = 'loss'
        MCCallBack    = callbacks.ModelCheckpoint(filepath=MCFile, monitor=Monitor, save_best_only=True, save_weights_only=True, verbose=0, mode='auto', save_freq='epoch', options=None)
        LRCallBack    = customReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=500, mode='auto', min_delta=1.e-4, cooldown=0, min_lr=1.e-8, verbose=1)
        TBCallBack    = callbacks.TensorBoard(log_dir=InputData.TBCheckpointFldr, histogram_freq=0, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
        
        CallBacksList = [ESCallBack, MCCallBack, LRCallBack, TBCallBack]
        #CallBacksList = [ESCallBack, MCCallBack, TBCallBack]

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