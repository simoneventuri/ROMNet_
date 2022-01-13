import tensorflow                             as tf
from tensorflow.python.util.tf_export     import keras_export



def get_learning_rate(lr, decay):
    if decay is None:
        return lr
    return {
        "exponential":  tf.keras.optimizers.schedules.ExponentialDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "inverse_time": tf.keras.optimizers.schedules.InverseTimeDecay(
            lr, decay[1], decay[2], staircase=True
        ),
        "polynomial":   tf.keras.optimizers.schedules.PolynomialDecay(
            lr, decay[1], end_learning_rate=1.e-2*lr
        )
    }[decay[0]]



def get_optimizer(InputData):

    LearningRate = get_learning_rate(InputData.LR, InputData.LRDecay)
    print('LearningRate = ', LearningRate)

    MTD = InputData.Optimizer
    if (MTD == 'adadelta'):  # A SGD method based on adaptive learning rate
        opt_name = 'Adadelta'
        args     = {'learning_rate': LearningRate, 'rho': 0.95, 'epsilon': InputData.epsilon, 'name': 'Adadelta'}

    elif (MTD == 'adagrad'):
        opt_name = 'Adagrad'
        args     = {'learning_rate': LearningRate, 'initial_accumulator_value': 0.1, 'epsilon': InputData.epsilon, 'name': "Adagrad"}
    
    elif (MTD == 'adam'):    # A SGD method based on adaptive estimation of first-order and second-order moments
        opt_name = 'Adam'
        args     = {'learning_rate': LearningRate, 'beta_1': InputData.OptimizerParams[0], 'beta_2': InputData.OptimizerParams[1], 'epsilon': InputData.OptimizerParams[2], 'amsgrad': False, 'name': 'Adam'}
    
    elif (MTD == 'adamax'):  # Variant of Adam algorithm based on the infinity norm.
        opt_name = 'Adam'
        args     = {'learning_rate': LearningRate, 'beta_1': InputData.OptimizerParams[0], 'beta_2': InputData.OptimizerParams[1], 'epsilon': InputData.OptimizerParams[2], 'name': 'Adamax'}
    
    elif (MTD == 'ftrl'):
        opt_name = 'Ftrl'
        args     = {'learning_rate': LearningRate, 'learning_rate_power': -0.5, 'initial_accumulator_value': 0.1, 'l1_regularization_strength': kW1, 'l2_regularization_strength': kW2, 'l2_regularization_strength': 0.0, 'beta': 0.0, 'name': 'Ftrl'}
    
    elif (MTD == 'nadam'):   # Variant of Adam algorithm with Nesterov momentum
        opt_name = 'Nadam'
        args     = {'learning_rate': LearningRate, 'beta_1': InputData.OptimizerParams[0], 'beta_2': InputData.OptimizerParams[1], 'epsilon': InputData.OptimizerParams[2], 'name': 'Nadam'}
    
    elif (MTD == 'rmsprop'):
        opt_name = 'RMSprop'
        args     = {'learning_rate': LearningRate, 'rho': 0.9, 'momentum': InputData.OptimizerParams[0], 'epsilon': InputData.OptimizerParams[1], 'centered': False, 'name': 'RMSprop'}
    
    elif (MTD == 'sgd'):
        opt_name = 'SGD'
        args     = {'learning_rate': LearningRate, 'name': "SGD", 'momentum': InputData.OptimizerParams[0], 'nesterov': True}
    
    else:
        raise ValueError("Unrecognized Optimizer!   InputData.Optimizer = ", InputData.Optimizer)


    opt_class = getattr(tf.keras.optimizers, opt_name)



    class SoftAttention_Optimizer(opt_class):
            
        def _get_gradients(self, tape, loss, var_list, grad_loss=None):
            """Called in `minimize` to compute gradients from loss."""
            
            grads = tape.gradient(loss, var_list, grad_loss)
            
            for ivar in range(len(grads)-1,len(grads)-3,-1):
                grads[ivar] = - grads[ivar] 

            return list(zip(grads, var_list))

    try:
        exist_flg = (InputData.Callbacks['weighted_loss']['name'] == 'SoftAttention')
        print('    [ROMNet]: Initializing Soft-Attention Mechanism for the ', opt_name,' Optimizer')
    except:
        exist_flg = False

    if (exist_flg):
        opt_ini = SoftAttention_Optimizer(**args)
    else:
        opt_ini = opt_class(**args)



    return opt_ini
