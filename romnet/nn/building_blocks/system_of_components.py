import numpy                                  as np
import tensorflow                             as tf
import h5py

from tensorflow.keras                     import regularizers
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras              import backend
from tensorflow.python.keras.engine       import base_layer_utils
from tensorflow.python.keras.utils        import tf_utils
from tensorflow.python.util.tf_export     import keras_export

from .component                           import Component


#=======================================================================================================================================
class System_of_Components(object):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, system_name, norm_input, layers_dict=[], layer_names_dict=[]):
        
        self.name                          = system_name

        if ('_' in self.name):
            self.type, self.idx            = self.name.split('_')
            self.idx                       = int(self.idx)
        else:           
            self.idx                       = 1
            self.type                      = self.name 
                  
        self.structure                     = InputData.structure

        self.input_vars                    = []
        for input_vars_ in InputData.input_vars[self.name].values():
            self.input_vars               += input_vars_
        self.input_vars                    = list(set(self.input_vars))
        self.n_inputs                      = len(self.input_vars)
        
        if (norm_input is not None):
            self.norm_input                = norm_input[self.input_vars]

        self.output_vars                   = InputData.output_vars
        self.n_outputs                     = len(self.output_vars)


        if (self.type == 'FNN'):    
            self.call                      = self.call_fnn
            
            self.system_post_layer_flg     = None


        elif (self.type == 'DeepONet'):    
            self.call                      = self.call_deeponet

            self.n_branches                = len([name for name in self.structure[self.name].keys() if 'Branch' in name])
            self.branch_vars               = InputData.input_vars[self.name]['Branch']

            try:
                self.n_rigids              = len([name for name in self.structure[self.name].keys() if 'Rigid' in name]) 
                self.rigid_vars            = InputData.input_vars[self.name]['Rigid']
            except:
                self.n_rigids              = 0

            self.n_trunks                  = len([name for name in self.structure[self.name].keys() if 'Trunk' in name])
            self.trunk_vars                = InputData.input_vars[self.name]['Trunk']

            try:
                self.branch_to_trunk       = InputData.branch_to_trunk[self.name]
                if (self.branch_to_trunk  == 'stacked'):
                    self.branch_to_trunk   = [0]*self.n_branches
                elif (self.branch_to_trunk == 'unstacked'):
                    self.branch_to_trunk   = np.arange(self.n_branches)
            except:    
              self.branch_to_trunk        = [0]*self.n_branches
            print("[ROMNet - system_of_components.py   ]:     Mapping Branch-to-Trunk (i.e., self.branch_to_trunk Object): ", self.branch_to_trunk) 

            try:
                self.system_post_layer_flg = InputData.system_post_layer_flg[self.name]
            except:
                self.system_post_layer_flg = None

            try:
                self.transfered_model      = InputData.transfered_model
            except:
                self.transfered_model      = None
                
        print("[ROMNet - system_of_components.py   ]:     Constructing System of Components: " + self.name) 

        


        # Iterating over Components
        self.components   = {}
        self.branch_names = []
        self.rigid_names  = []
        self.trunk_names  = []
        for component_name in self.structure[self.name]:

            if  ('Branch' in component_name):
                self.branch_names.append(component_name)
            elif ('Rigid' in component_name):
                self.rigid_names.append(component_name)
            elif ('Trunk' in component_name):
                self.trunk_names.append(component_name)

            if (not component_name in layers_dict[self.name]):
                layers_dict[self.name][component_name]      = {}
                layer_names_dict[self.name][component_name] = {}
            self.components[component_name]                 = Component(InputData, self.name, component_name, self.norm_input, layers_dict=layers_dict, layer_names_dict=layer_names_dict)


        # Adding Post Correlating / Scaling / Shifting Layer (If Needed)
        if (self.system_post_layer_flg):
            
            self.post_layer_vec = []

            if (self.system_post_layer_flg == 'correlation'):
                i_branch = 0
                self.post_layer_vec.append(system_post_layer(self.system_post_layer_flg, self.name, self.n_branches, self.transfered_model))

            else: 
                for i_branch in range(n_branches):
                    self.post_layer_vec.append(system_post_layer(self.system_post_layer_flg, self.name, i_branch, self.transfered_model))

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_fnn(self, inputs, layers_dict, training=False):

        y = self.components['FNN'].call(inputs, layers_dict, None, None, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_deeponet(self, inputs, layers_dict, training):

        inputs_branch, inputs_trunk = inputs


        if (self.n_rigids > 0):
            for i_rigid in range(self.n_rigids):
                rigid_name = self.rigid_names[i_rigid]
                rigid      = self.components[rigid_name].call(inputs_branch, layers_dict, None, None, training=training)
            splits  = tf.split(rigid, num_or_size_splits=[1]*self.n_trunks, axis=1)
            #splits  = tf.split(rigid, num_or_size_splits=[1]*self.n_trunks*2, axis=1)
            #shift   = splits[0:self.n_trunks]
            #stretch = [None]*self.n_trunks
            #stretch = splits[self.n_trunks:self.n_trunks*2]
            stretch = splits[0:self.n_trunks]
            shift   = [None]*self.n_trunks
        else:
            shift   = [None]*self.n_trunks
            stretch = [None]*self.n_trunks



        y_trunk_vec = []
        for i_trunk in range(self.n_trunks): 
            trunk_name  = self.trunk_names[i_trunk]
            y_trunk_vec.append(self.components[trunk_name].call(inputs_trunk, layers_dict, shift[i_trunk], stretch[i_trunk], training=training))


        y_branch_vec = []
        output_vec   = []
        for i_branch in range(self.n_branches): 
            i_trunk     = self.branch_to_trunk[i_branch]
            branch_name = self.branch_names[i_branch] 
            y           = self.components[branch_name].call(inputs_branch, layers_dict, None, None, training=training)


            output_dot = Dot_Add(axes=1)([y, y_trunk_vec[i_trunk]])            


            if (self.system_post_layer_flg == 'shift') or (self.system_post_layer_flg == 'linear'):
                output_vec.append( self.post_layer_vec[i_branch](output_dot, training=training) )
            else:
                output_vec.append( output_dot )
                

        if (self.n_branches > 1):
            output_concat = tf.keras.layers.Concatenate(axis=1)(output_vec)
        else:
            output_concat = output_vec[0]


        if (self.system_post_layer_flg == 'correlation'):
            output_final = self.post_layer_vec[0](output_concat, training=training)
        else:
            output_final = output_concat


        return output_final

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================



#=======================================================================================================================================
@keras_export('keras.layers.Dot_Add')
class Dot_Add(_Merge):
    """Layer that computes a dot product between samples in two tensors.
    E.g. if applied to a list of two tensors `a` and `b` of shape
    `(batch_size, n)`, the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.
    >>> x = np.arange(10).reshape(1, 5, 2)
    >>> print(x)
    [[[0 1]
        [2 3]
        [4 5]
        [6 7]
        [8 9]]]
    >>> y = np.arange(10, 20).reshape(1, 2, 5)
    >>> print(y)
    [[[10 11 12 13 14]
        [15 16 17 18 19]]]
    >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
    <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
    array([[[260, 360],
                    [320, 445]]])>
    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> dotted = tf.keras.layers.Dot(axes=1)([x1, x2])
    >>> dotted.shape
    TensorShape([5, 1])
    """

    def __init__(self, axes, normalize=False, **kwargs):
        """Initializes a layer that computes the element-wise dot product.
            >>> x = np.arange(10).reshape(1, 5, 2)
            >>> print(x)
            [[[0 1]
                [2 3]
                [4 5]
                [6 7]
                [8 9]]]
            >>> y = np.arange(10, 20).reshape(1, 2, 5)
            >>> print(y)
            [[[10 11 12 13 14]
                [15 16 17 18 19]]]
            >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
            <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
            array([[[260, 360],
                            [320, 445]]])>
        Args:
            axes: Integer or tuple of integers,
                axis or axes along which to take the dot product. If a tuple, should
                be two integers corresponding to the desired axis from the first input
                and the desired axis from the second input, respectively. Note that the
                size of the two selected axes must match.
            normalize: Whether to L2-normalize samples along the
                dot product axis before taking the dot product.
                If set to True, then the output of the dot product
                is the cosine proximity between the two samples.
            **kwargs: Standard layer keyword arguments.
        """
        super(Dot_Add, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError(
                        'Invalid type for argument `axes`: it should be '
                        f'a list or an int. Received: axes={axes}')
            if len(axes) != 2:
                raise ValueError(
                        'Invalid format for argument `axes`: it should contain two '
                        f'elements. Received: axes={axes}')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError(
                        'Invalid format for argument `axes`: list elements should be '
                        f'integers. Received: axes={axes}')
        self.axes              = axes
        self.normalize         = normalize
        self.supports_masking  = True
        self._reshape_required = False


    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], tuple) or len(input_shape) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on a list of 2 inputs. '
                    f'Received: input_shape={input_shape}')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        
        self.n_branch = shape1[axes[0]]
        self.n_trunk  = shape2[axes[1]]
        if (self.n_trunk < self.n_branch - 2) or (self.n_trunk > self.n_branch):
            raise ValueError(
                    'Incompatible input shapes: '
                    f'axis values {shape1[axes[0]]} (at axis {axes[0]}) != '
                    f'{shape2[axes[1]]*2} (at axis {axes[1]}). '
                    f'Full input shapes: {shape1}, {shape2}')


    def _merge_function(self, inputs):
        base_layer_utils.no_ragged_support(inputs, self.name)
        if len(inputs) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on exactly 2 inputs. '
                    f'Received: inputs={inputs}')
        x1         = inputs[1]
        if (self.n_trunk == self.n_branch):
            x2 = inputs[0]
        elif (self.n_trunk == self.n_branch-1):
            x2, x3 = tf.split(inputs[0], num_or_size_splits=[self.n_trunk, 1], axis=1)
        elif (self.n_trunk == self.n_branch-2):
            x2, x3, x4 = tf.split(inputs[0], num_or_size_splits=[self.n_trunk, 1, 1], axis=1)

        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % backend.ndim(x1), self.axes % backend.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % backend.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = tf.linalg.l2_normalize(x1, axis=axes[0])
            x2 = tf.linalg.l2_normalize(x2, axis=axes[1])

        if (self.n_trunk == self.n_branch):
            output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True)
        elif (self.n_trunk == self.n_branch-1):
            output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True) + x3
        elif (self.n_trunk == self.n_branch-2):
            output = tf.math.reduce_sum( tf.math.multiply(x1, x2), axis=1, keepdims=True)*x4 + x3

        return output


    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                    'A `Dot` layer should be called on a list of 2 inputs. '
                    f'Received: input_shape={input_shape}')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)


    def compute_mask(self, inputs, mask=None):
        return None


    def get_config(self):
        config = {
                'axes': self.axes,
                'normalize': self.normalize,
        }
        base_config = super(Dot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#=======================================================================================================================================



#=======================================================================================================================================
class bias_layer(tf.keras.layers.Layer):

    def __init__(self, b0, layer_name):
        super(bias_layer, self).__init__(name=layer_name)
        self.b0   = b0

    def build(self, input_shape):
        b_ini     = tf.keras.initializers.constant(value=self.b0)
        self.bias = self.add_weight('bias',
                                    shape       = input_shape[1:],
                                    initializer = b_ini,
                                    trainable   = True)

    def call(self, x):
        return x + self.bias

#=======================================================================================================================================



#===================================================================================================================================
class DenseVariational_Mod(tf.keras.layers.Layer):
    """Dense layer with random `kernel` and `bias`.
    This layer uses variational inference to fit a "surrogate" posterior to the
    distribution over both the `kernel` matrix and the `bias` terms which are
    otherwise used in a manner similar to `tf.keras.layers.Dense`.
    This layer fits the "weights posterior" according to the following generative
    process:
    ```none
    [K, b] ~ Prior()
    M = matmul(X, K) + b
    Y ~ Likelihood(M)
    ```
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self,
                 units,
                 kernel0,
                 bias0,
                 make_posterior_fn,
                 make_prior_fn,
                 kl_weight=None,
                 kl_use_exact=False,
                 activation=None,
                 use_bias=True,
                 activity_regularizer=None,
                 **kwargs):
        """Creates the `DenseVariational` layer.
        Args:
            units: Positive integer, dimensionality of the output space.
            make_posterior_fn: Python callable taking `tf.size(kernel)`,
                `tf.size(bias)`, `dtype` and returns another callable which takes an
                input and produces a `tfd.Distribution` instance.
            make_prior_fn: Python callable taking `tf.size(kernel)`, `tf.size(bias)`,
                `dtype` and returns another callable which takes an input and produces a
                `tfd.Distribution` instance.
            kl_weight: Amount by which to scale the KL divergence loss between prior
                and posterior.
            kl_use_exact: Python `bool` indicating that the analytical KL divergence
                should be used rather than a Monte Carlo approximation.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation")..
            **kwargs: Extra arguments forwarded to `tf.keras.layers.Layer`.
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseVariational_Mod, self).__init__(activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)

        self.kernel0            = kernel0
        self.bias0              = bias0
        self.pars0              = tf.concat([self.kernel0, self.bias0], 0)

        self._make_posterior_fn = make_posterior_fn
        self._make_prior_fn     = make_prior_fn
        self._kl_divergence_fn  = _make_kl_divergence_penalty(kl_use_exact, weight=kl_weight)

        self.activation         = tf.keras.activations.get(activation)
        self.use_bias           = use_bias
        self.supports_masking   = False
        self.input_spec         = tf.keras.layers.InputSpec(min_ndim=2)

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point dtype %s' % (dtype,))

        input_shape = tf.TensorShape(input_shape)
        last_dim    = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `DenseVariational` should be defined. Found `None`.')
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        self._posterior = self._make_posterior_fn(
                last_dim * self.units,
                self.units if self.use_bias else 0,
                dtype)

        #pars0 = tf.zeros(tf.int32(last_dim * self.units + self.units), dtype=dtype)
        self._prior = self._make_prior_fn(
                last_dim * self.units,
                self.units if self.use_bias else 0,
                self.pars0, dtype)

        self.built = True

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs):
        dtype  = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        inputs = tf.cast(inputs, dtype, name='inputs')

        q = self._posterior(inputs)
        r = self._prior(inputs)
        self.add_loss(self._kl_divergence_fn(q, r))

        w = tf.convert_to_tensor(value=q)
        prev_units = self.input_spec.axes[-1]
        if self.use_bias:
            split_sizes = [prev_units * self.units, self.units]
            kernel, bias = tf.split(w, split_sizes, axis=-1)
        else:
            kernel, bias = w, None

        kernel = tf.reshape(kernel, shape=tf.concat([
                tf.shape(kernel)[:-1],
                [prev_units, self.units],
        ], axis=0))
        outputs = tf.matmul(inputs, kernel)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

        if self.activation is not None:
            outputs = self.activation(outputs)  # pylint: disable=not-callable

        return outputs

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================



#===================================================================================================================================
def _make_kl_divergence_penalty(
        use_exact_kl=False,
        test_points_reduce_axis=(),  # `None` == "all"; () == "none".
        test_points_fn=tf.convert_to_tensor,
        weight=None):
    """Creates a callable computing `KL[a,b]` from `a`, a `tfd.Distribution`."""

    if use_exact_kl:
        kl_divergence_fn = kullback_leibler.kl_divergence
    else:
        def kl_divergence_fn(distribution_a, distribution_b):
            z = test_points_fn(distribution_a)
            return tf.reduce_mean(
                    distribution_a.log_prob(z) - distribution_b.log_prob(z),
                    axis=test_points_reduce_axis)

    # Closure over: kl_divergence_fn, weight.
    def _fn(distribution_a, distribution_b):
        """Closure that computes KLDiv as a function of `a` as in `KL[a, b]`."""
        with tf.name_scope('kldivergence_loss'):
            kl = kl_divergence_fn(distribution_a, distribution_b)
            if weight is not None:
                kl = tf.cast(weight, dtype=kl.dtype) * kl
            # Losses appended with the model.add_loss and are expected to be a single
            # scalar, unlike model.loss, which is expected to be the loss per sample.
            # Therefore, we reduce over all dimensions, regardless of the shape.
            # We take the sum because (apparently) Keras will add this to the *post*
            # `reduce_sum` (total) loss.
            # TODO(b/126259176): Add end-to-end Keras/TFP test to ensure the API's
            # align, particularly wrt how losses are aggregated (across batch
            # members).
            return tf.reduce_sum(kl, name='batch_total_kl_divergence')

    return _fn

#===================================================================================================================================



#=======================================================================================================================================
def system_post_layer(system_post_layer_flg, system_name, i_out, transfered_model):

    if (system_post_layer_flg == 'correlation'):
        layer_name = system_name + '-Post_Correlation'
        
        if (transfered_model is not None):
            W0     = transfered_model.get_layer(layer_name).kernel.numpy()
            b0     = transfered_model.get_layer(layer_name).bias.numpy()
            W_ini  = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini  = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            W_ini  = 'he_normal'
            b_ini  = 'zeros'
        out_layer  = tf.keras.layers.Dense(units              = i_out,
                                           activation         = 'linear',
                                           use_bias           = True,
                                           kernel_initializer = W_ini,
                                           bias_initializer   = b_ini,
                                           name               = layer_name)

    elif (system_post_layer_flg == 'linear'):
        layer_name = system_name + '-Post_Linear_' + str(i_out+1)
        
        if (transfered_model is not None):
            W0     = transfered_model.get_layer(layer_name).kernel.numpy()
            b0     = transfered_model.get_layer(layer_name).bias.numpy()
            W_ini  = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini  = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
        else:
            W_ini  = 'he_normal'
            b_ini  = 'zeros'
        out_layer  = tf.keras.layers.Dense(units              = 1,
                                           activation         = 'linear',
                                           use_bias           = True,
                                           kernel_initializer = W_ini,
                                           bias_initializer   = b_ini,
                                           name               = layer_name)
    
    elif (system_post_layer_flg == 'shift'):
        layer_name = system_name + '-Post_Shift_' + str(i_out+1)

        b0 = 0
        if (transfered_model is not None): 
            b0 = transfered_model.get_layer(layer_name).bias.numpy()[0]
        out_layer = bias_layer(b0=b0, layer_name=layer_name)

    return out_layer

#=======================================================================================================================================