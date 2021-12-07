import sys
import itertools
import numpy                                  as np
from pathlib                              import Path

import tensorflow                             as tf
import tensorflow.keras.backend               as K

from tqdm                                 import tqdm
from tensorflow.python.eager              import backprop
from tensorflow.python.util.tf_export     import keras_export



#=======================================================================================================================================
@keras_export('keras.callbacks.ConstantWeightsAdapter')
class ConstantWeightsAdapter(tf.keras.callbacks.Callback):
    """Base class for adaptive loss weights techniques."""

    def __init__(
        self,
        pde_loss_weights=None,
        data_generator=None,
        freq=1,
        log_dir=None,
        **kwargs
    ):
        super(ConstantWeightsAdapter, self).__init__()

        print('pde_loss_weights = ', pde_loss_weights)
        self.name             = None
        self.data_generator   = data_generator
        self.pde_loss_weights = pde_loss_weights
        self.freq             = np.clip(freq, 1, None) if freq is not None else np.inf
        self.log_dir          = log_dir

    def on_train_begin(self, logs={}):
        self.init_weights()

    def on_epoch_end(self, epoch, logs={}):
        self.print_weights(epoch, logs)

    def init_weights(self):
        if self.pde_loss_weights is None:
            print("[ROMNet]   No `pde_loss_weights` have been provided. The initial value of 1 will be applied.")
            _pde_loss_weights = dict( zip(self.model.data_ids, [1.]*len(self.model.data_ids)) )
        elif isinstance(self.pde_loss_weights, (list,tuple)):
            _pde_loss_weights = dict( zip(self.model.data_ids, self.pde_loss_weights) )
        elif isinstance(self.pde_loss_weights, dict):
            _pde_loss_weights = self.pde_loss_weights
        else:
            utils.raise_value_err("`pde_loss_weights` passed into `{}` "
                "has to be a list or a dictionary:\n"
                "  - 'ics': weight for initial condition pts loss;\n"
                "  - 'pts': weight for anchor pts loss;\n"
                "  - 'res': weight for residual loss.".format(self.name))
        self.model.pde_loss_weights = _pde_loss_weights
        
        if self.log_dir is not None:
            self.logs = { k: tf.summary.create_file_writer( self.log_dir + '/scalars/pde_loss_weights/' + k ) for k in self.model.pde_loss_weights }



    def print_weights(self, epoch, logs):
        log_weights = { k+'_loss_weight': v for k,v in self.model.pde_loss_weights.items() }
        logs.update(log_weights)
        if self.log_dir is not None:
            for k, log in self.logs.items():
                with log.as_default():
                    tf.summary.scalar(name='epoch_'+k+'_loss_weight', \
                        data=self.model.pde_loss_weights[k], step=epoch)


    def check_weights(self):
        weights = self.model.pde_loss_weights.values()
        if any(tf.math.is_nan(w) for w in weights) \
            or any(tf.math.is_inf(w) for w in weights):
            print("")
            utils.warning(
                "One or multiple `pde_loss_weights` evaluated in `{}` class "
                "are `inf` or `nan`.".format(self.name)
            )
            utils.print_submain("Here the `pde_loss_weights` values:")
            for k,v in self.model.pde_loss_weights.items():
                utils.print_submain("  - '{}': {}".format(k,v))
            utils.print_submain("The training will be stopped.")
            self.model.stop_training = True

#=======================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.callbacks.SoftAttention')
class SoftAttention(tf.keras.callbacks.Callback):
    """Base class for adaptive loss weights techniques."""

    def __init__(
        self,
        pde_loss_weights=None,
        data_generator=None,
        freq=1,
        log_dir=None,
        n_train_tot=None,
        loss_weights0=None,    
        save_dir=None,
        shape_1=1,
        **kwargs
    ):
        super(SoftAttention, self).__init__()

        self.name             = 'soft_attention'
        self.pde_loss_weights = pde_loss_weights
        self.data_generator   = data_generator
        self.freq             = np.clip(freq, 1, None) if freq is not None else np.inf
        self.log_dir          = log_dir
        self.n_train_tot      = n_train_tot
        self.loss_weights0    = loss_weights0    
        self.save_dir         = save_dir
        self.shape_1          = shape_1


    def on_train_begin(self, logs={}):
        self.init_weights()

        self.add_softattention_weights()
        
        path = Path( self.save_dir+'/Model/LossWeights' )
        path.mkdir(parents=True, exist_ok=True)

        path = Path( self.save_dir+'/Training/LossWeights' )
        path.mkdir(parents=True, exist_ok=True)
        self.save_softattention_weights(self.save_dir+'/Model/LossWeights/', 'Initial')

        print('self.save_dir = ', self.save_dir)


    def on_epoch_end(self, epoch, logs={}):
        if (epoch % self.freq == 0):
            self.save_softattention_weights(self.save_dir+'/Training/LossWeights/', str(epoch))


    def on_train_end(self, logs={}):
        self.save_softattention_weights(self.save_dir+'/Model/LossWeights/', 'Final')


    def add_softattention_weights(self):
        self.model.attention_mask = []
        for key, value in self.n_train_tot.items():
            weight0 = self.loss_weights0[key]
            if (key == 'ics'):
                self.model.attention_mask.append(tf.Variable(np.ones((value,self.shape_1))*weight0, name='att_mask_'+key, trainable=True))
                #self.model.attention_mask.append(tf.Variable(np.random.rand(value,self.shape_1)*weight0, name='att_mask_'+key, trainable=True))
            elif (key == 'res'):
                self.model.attention_mask.append(tf.Variable(np.ones((value,self.shape_1))*weight0, name='att_mask_'+key, trainable=True))
                #self.model.attention_mask.append(tf.Variable(np.random.rand(value,self.shape_1)*weight0, name='att_mask_'+key, trainable=True))


    def save_softattention_weights(self, fld_name, suffix):
        i=0    
        for key, value in self.n_train_tot.items():
            np.savetxt(fld_name+key+'_'+suffix+'.csv', self.model.attention_mask[i].numpy(), delimiter=",")
            i+=1


    def init_weights(self):
        if self.pde_loss_weights is None:
            print("[ROMNet]   No `pde_loss_weights` have been provided. The initial value of 1 will be applied.")
            _pde_loss_weights = dict( zip(self.model.data_ids, [1.]*len(self.model.data_ids)) )
        elif isinstance(self.pde_loss_weights, (list,tuple)):
            _pde_loss_weights = dict( zip(self.model.data_ids, self.pde_loss_weights) )
        elif isinstance(pde_loss_weights, dict):
            _pde_loss_weights = self.pde_loss_weights
        else:
            utils.raise_value_err("`pde_loss_weights` passed into `{}` "
                "has to be a list or a dictionary:\n"
                "  - 'ics': weight for initial condition pts loss;\n"
                "  - 'pts': weight for anchor pts loss;\n"
                "  - 'res': weight for residual loss.".format(self.name))
        self.model.pde_loss_weights = _pde_loss_weights
        
        if self.log_dir is not None:
            self.logs = { k: tf.summary.create_file_writer( self.log_dir + '/scalars/pde_loss_weights/' + k ) for k in self.model.pde_loss_weights }

#=======================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.callbacks.EmpiricalWeightsAdapter')
class EmpiricalWeightsAdapter(ConstantWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on an empirical learning
    rate annealing algorithm.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    def __init__(
        self,
        pde_loss_weights=None,
        data_generator=None,
        freq=1000,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
        method='mean',
        **kwargs
    ):
        super(EmpiricalWeightsAdapter, self).__init__(
            pde_loss_weights=pde_loss_weights,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir
        )

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())
        self.loss_gradients = {}
        self.dataset = {}
        self.max_samples = max_samples

        self.grads_eval = self.mean if method == 'mean' else self.max
        self._update = True

    def on_train_begin(self, logs={}):
        self.init_weights()
        if 'res' not in self.model.data_ids:
            utils.warning(
                "Network `{}` hasn't any residual loss to be evaluated: "
                "`{}` can't be applied.".format(self.model.name, self.name)
            )
            self._update = False

    def on_epoch_begin(self, epoch, logs={}):
        if self._update and epoch % self.freq == 0:
            self.update()

    def update(self):
        self.sample_dataset()
        self.eval_loss_gradients()
        self.update_loss_weights()
        self.check_weights()

    # Sampling data
    ###########################################################################
    def sample_dataset(self):
        for data_id, dset in self.data_generator.data.items():
            n_samples = utils.get_cardinality(dset)
            indeces = np.random.permutation(n_samples)
            if n_samples > self.max_samples:
                indeces = np.random.choice(
                    indeces, self.max_samples, replace=False
                )
            self.dataset[data_id] = self.shuffle_data(dset, indeces)

    def shuffle_data(self, dset, indeces):
        _dset = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                _dset.append([tf.convert_to_tensor(i[indeces], \
                    dtype=K.floatx()) for i in xy])
            else:
                _dset.append(tf.convert_to_tensor(xy[indeces], \
                    dtype=K.floatx()))
        return _dset

    # Handling loss gradients
    ###########################################################################
    def eval_loss_gradients(self):
        pred, losses = {}, {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                pred[data_id], losses[data_id] = self.get_pred_loss(
                    *self.dataset[data_id], data_id=data_id
                )
        for data_id in self.model.data_ids:
            self.loss_gradients[data_id] = tape.gradient(
                losses[data_id], self.model.trainable_variables
            )

    def get_pred_loss(self, x, y, data_id):
        if data_id == 'res':
            y_pred = self.model.residual(x, training=False)
            if isinstance(y_pred, (list,tuple)):
                y = [ tf.zeros_like(y_pred_i) for y_pred_i in y_pred ]
            else:
                y = tf.zeros_like(y_pred)
        else:
            y_pred = self.model(x, training=False)
        loss = self.model.compiled_loss[data_id](y, y_pred)
        return y_pred, loss

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.pde_loss_weights.items():
            if data_id == 'res':
                continue
            weight = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.pde_loss_weights[data_id]
            self.model.pde_loss_weights[data_id] = weight.numpy()

    def eval_loss_weights(self):
        # Get the maximum/mean gradient value from residual loss (see `method')
        grad_res = self.grads_eval(self.loss_gradients['res'])
        new_weights = {}
        for data_id, grad in self.loss_gradients.items():
            if data_id == 'res':
                continue
            # Get the mean of the gradients for each remaining loss term
            grads_mean = self.mean(grad)
            # Update the loss weights
            new_weights[data_id] = grad_res / grads_mean \
                * self.model.pde_loss_weights['res']
        return new_weights

    def max(self, x):
        return tf.reduce_max([tf.reduce_max(tf.math.abs(i)) for i in x])

    def mean(self, x):
        return tf.reduce_mean([tf.reduce_mean(tf.math.abs(i)) for i in x])

#=======================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.callbacks.InverseDirichletWeightsAdapter')
class InverseDirichletWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Inverse-Dirichlet
    weighting."""

    def __init__(
        self,
        pde_loss_weights=None,
        data_generator=None,
        freq=1000,
        alpha=0.8,
        log_dir=None,
        max_samples=20000,
        method='std',
        **kwargs
    ):
        super(InverseDirichletWeightsAdapter, self).__init__(
            pde_loss_weights=pde_loss_weights,
            data_generator=data_generator,
            freq=freq,
            alpha=alpha,
            log_dir=log_dir,
            max_samples=max_samples,
            **kwargs
        )

        self.grads_eval = np.var if method == 'var' else np.std

    def on_train_begin(self, logs={}):
        self.init_weights()

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self):
        new_weights = self.eval_loss_weights()
        for data_id, loss in self.model.pde_loss_weights.items():
            weight = self.alpha * new_weights[data_id] + \
                (1. - self.alpha) * self.model.pde_loss_weights[data_id]
            self.model.pde_loss_weights[data_id] = weight.numpy()

    def eval_loss_weights(self):
        sigmas = {}
        for data_id, grad in self.loss_gradients.items():
            sigmas[data_id] = self.grads_eval(
                np.concatenate([ g.numpy().flatten() for g in grad ])
            )
        sigma_max = max(sigmas.values())
        return {data_id: sigma_max/sigma for data_id, sigma in sigmas.items()}


@keras_export('keras.callbacks.NTKWeightsAdapter')
class NTKWeightsAdapter(EmpiricalWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Neural Tangent Kernel
    (NTK) theory.

    See:
        Sifan Wang, Xinling Yu, Paris Perdikaris
        `When and why PINNs fail to train: A neural tangent kernel perspective`
        https://arxiv.org/abs/2007.14527"""

    def __init__(
        self,
        pde_loss_weights=None,
        data_generator=None,
        freq=None,
        log_dir=None,
        max_samples=1000,
        **kwargs
    ):
        super(NTKWeightsAdapter, self).__init__(
            pde_loss_weights=pde_loss_weights,
            data_generator=data_generator,
            freq=freq,
            log_dir=log_dir,
            max_samples=max_samples,
            **kwargs
        )
        self.pred_fn = {}

    def on_train_begin(self, logs={}):
        self.init_weights()
        self.set_pred_fn()

    def update(self):
        self.sample_dataset()
        ntk = self.get_ntk()
        self.update_loss_weights(ntk)
        self.check_weights()

    # Handling predictions
    ###########################################################################
    def set_pred_fn(self):
        for data_id in self.model.data_ids:
            if data_id == 'res':
                _pred_fn = self.model.residual
            else:
                _pred_fn = self.model.call
            self.pred_fn[data_id] = _pred_fn

    # Handling loss gradients
    ###########################################################################
    def flatten(self, x):
        # Flatten array or list of arrays into 1D vector
        if isinstance(x, (list,tuple)):
            return list(itertools.chain(
                *[tf.split(tf.reshape(i,[-1]), tf.size(i).numpy()) for i in x]
            ))
        else:
            return tf.split(tf.reshape(x,[-1]), tf.size(x).numpy())

    def get_ntk(self):
        pred = {}
        with backprop.GradientTape(persistent=True) as tape:
            for data_id in self.model.data_ids:
                x = self.dataset[data_id][0]
                pred[data_id] = self.flatten(
                    self.pred_fn[data_id](
                        self.dataset[data_id][0], training=False
                    )
                )
        return self.eval_diag_ntk(pred, tape)

    def eval_diag_ntk(self, pred, tape):
        utils.print_submain("From `NTKWeightsAdapter` callback:")
        ntk = {}
        for data_id, pred_i in pred.items():
            utils.print_submain("  - Evaluating NTK matrix for `{}` data "
                "points".format(data_id))
            ta = tf.TensorArray(
                dtype=K.floatx(),
                size=0,
                dynamic_size=True,
                clear_after_read=False
            )
            for i, elem in enumerate(tqdm(pred_i, file=sys.stdout)):
                grad = tape.gradient(
                    elem,
                    self.model.trainable_variables,
                    unconnected_gradients='zero'
                )
                tr = tf.add_n([tf.reduce_sum(tf.multiply(g, g)) for g in grad])
                ta = ta.write(i, tr)
            ntk[data_id] = ta.stack()
        return ntk

    # Handling loss weights
    ###########################################################################
    def update_loss_weights(self, ntk):
        trace = {
            data_id: tf.reduce_sum(ntk_i).numpy() \
                for data_id, ntk_i in ntk.items()
        }
        norm_grad = tf.reduce_sum(list(trace.values())).numpy()
        for data_id in self.model.data_ids:
            self.model.pde_loss_weights[data_id] = norm_grad / trace[data_id]

#=======================================================================================================================================
