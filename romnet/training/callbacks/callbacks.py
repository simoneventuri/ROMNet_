import os
import sys
import time
import copy

import tensorflow as tf

from . import early_stopping
from . import learning_rate
from . import plotting
from . import variable
from . import weighted_loss
from . import base




#=======================================================================================================================================
def get_callback(model, InputData):

    #=======================================================================================================================================
    def _get_weighted_loss(model, pde_loss_weights, args):
        # Add loss weights to the network class model.net.add_loss_weights(model.loss_weights)
        # Manipulate arguments for `weighted_loss` callback

        if (args['data_generator'] is not None):
            data_generator = model.data.train
        else:
            data_generator = None

        try:
            if (args['shape_1'] is None) or (args['shape_1'] == '1'):
                args['shape_1'] = 1
            elif (args['shape_1'] == 'all'):
                args['shape_1'] = model.NOutputVars
        except:
            args['shape_1'] = 1

        args.update( { 'data_generator': data_generator, 'log_dir': model.TBCheckpointFldr, 'n_train_tot': model.data.n_train_tot, 'save_dir': model.PathToRunFld, 'pde_loss_weights': pde_loss_weights  } )
        
        Name = args.pop('name')

        WeightedLoss = getattr(weighted_loss, Name)

        return WeightedLoss(**args)

    #=======================================================================================================================================

    #=======================================================================================================================================
    def _get_lr_scheduler(args):
        # Manipulate arguments for `learning_rate` callback

        Name        = args.pop('name')
        LRScheduler = getattr(learning_rate, Name)

        return LRScheduler(**args)

    #=======================================================================================================================================

    #=======================================================================================================================================
    def _get_single_callback(model, Name, args):

        if   Name == "base":
            args.update( { 'PathToRunFld': InputData.PathToRunFld  } )
            return base.BaseLogger(**args)

        elif Name == "early_stopping":
            return tf.keras.callbacks.EarlyStopping(**args)

        elif Name == "model_ckpt":
            return tf.keras.callbacks.ModelCheckpoint( **args, filepath=model.PathToRunFld+"/Training/Params/{epoch:06d}.h5" )

        elif Name == "tensorboard":
            return tf.keras.callbacks.TensorBoard(**args, log_dir=model.TBCheckpointFldr)


        elif Name == "weighted_loss":
            return _get_weighted_loss(model, InputData.LossWeights, args)

        elif Name == "lr_scheduler":
            return _get_lr_scheduler(args)

        elif Name == "timer":
            return early_stopping.Timer(**args)

        elif Name == "lr_tracker":
            return learning_rate.LearningRateTracker( **args, file=model.PathToRunFld+'/Training/LR.csv', log_dir=model.TBCheckpointFldr )

        elif Name == "plotter":
            return plotting.Plotter( test_data=model.data.test, train_dir=model.PathToRunFld+'/Training/', **args )

        elif Name == "variable":
            return variable.VariableValue(**args)


        elif callable(Name):
            return Name

        elif (Name == None):
            return None
        else:
            raise ValueError("Unrecognized Callback!   Name = ", Name)

        #=======================================================================================================================================

    callback_list = [ _get_single_callback(model, Name, args) for Name, args in InputData.Callbacks.items() if args is not None ]

    if "weighted_loss" not in InputData.Callbacks:
        callback_list.append( _get_single_callback(model, "weighted_loss", {'name': 'ConstantWeightsAdapter', 'data_generator': None, 'pde_loss_weights': InputData.LossWeights}) )
    
    return callback_list

#=======================================================================================================================================

