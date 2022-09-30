import numpy                    as np
import tensorflow               as tf

from tensorflow.keras       import regularizers
from tensorflow.keras       import activations
from tensorflow.keras       import initializers
import tensorflow_probability   as tfp

from tensorflow_probability.python.layers        import util        as tfp_layers_util
from tensorflow_probability.python.distributions import normal      as normal_lib
from tensorflow_probability.python.distributions import independent as independent_lib

from ...training            import L1L2Regularizer


#=======================================================================================================================================
class Layer(object):
    

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
                 InputData, 
                 layer_type          = 'TF',
                 i_layer             = 1, 
                 n_layers            = 1, 
                 layer_name          = '', 
                 n_neurons           = 1, 
                 act_func            = 'linear', 
                 use_bias            = True, 
                 trainable_flg       = 'all', 
                 reg_coeffs          = [0., 0.],
                 transfered_model    = None):

        ### Weights L1 and L2 Regularization Coefficients 
        self.batch_size          = InputData.batch_size

        self.layer_type          = layer_type
        self.i_layer             = i_layer
        self.n_layers            = n_layers
        self.layer_name          = layer_name
        self.n_neurons           = n_neurons
        self.act_func            = act_func
        self.use_bias            = use_bias
        self.trainable_flg       = trainable_flg
        self.reg_coeffs          = reg_coeffs
        self.transfered_model    = transfered_model
        self.last_flg            = True if (i_layer >= n_layers-1) else False

        if (self.layer_type == 'TF'):
            self.build = self.build_TF
        elif (self.layer_type == 'TFP'):
            self.build = self.build_TFP_DenseLocal


    # ---------------------------------------------------------------------------------------------------------------------------
    def build_TF(self):

        # Parameters Initialization
        ### Biases L1 and L2 Regularization Coefficients 
        kW1 = self.reg_coeffs[0]
        kW2 = self.reg_coeffs[1]
        if (not self.last_flg):
            if (len(self.reg_coeffs) == 2):
                kb1 = self.reg_coeffs[0]
                kb2 = self.reg_coeffs[1]
            else:
                kb1 = self.reg_coeffs[2]
                kb2 = self.reg_coeffs[3]
        else:
            kb1 = 0.
            kb2 = 0.

        if (self.transfered_model is not None):
            W0    = self.transfered_model.get_layer(self.layer_name).kernel.numpy()
            b0    = self.transfered_model.get_layer(self.layer_name).bias.numpy()
            W_ini = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            W_reg = L1L2Regularizer(kW1, kW2, W0)
            b_reg = L1L2Regularizer(kb1, kb2, b0)
        else:
            W_ini = 'he_normal' if (self.act_func == 'relu') else 'glorot_normal'
            b_ini = 'zeros'
            W_reg = regularizers.l1_l2(l1=kW1, l2=kW2)
            b_reg = regularizers.l1_l2(l1=kb1, l2=kb2)

        # Constructing Kera Layer
        layer = tf.keras.layers.Dense(units              = self.n_neurons,
                                      activation         = self.act_func,
                                      use_bias           = self.use_bias,
                                      kernel_initializer = W_ini,
                                      bias_initializer   = b_ini,
                                      kernel_regularizer = W_reg,
                                      bias_regularizer   = b_reg,
                                      name               = self.layer_name)


        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer



    # ---------------------------------------------------------------------------------------------------------------------------
    def build_TFP_FlipOut(self):

        def nondefault_multivariate_normal_fn(dtype, shape, name, trainable, add_variable_fn):
            """Creates multivariate standard `Normal` distribution.
            Args:
            dtype: Type of parameter's event.
            shape: Python `list`-like representing the parameter's event shape.
            name: Python `str` name prepended to any created (or existing)
              `tf.Variable`s.
            trainable: Python `bool` indicating all created `tf.Variable`s should be
              added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
            add_variable_fn: `tf.get_variable`-like `callable` used to create (or
              access existing) `tf.Variable`s.
            Returns:
            Multivariate standard `Normal` distribution.å=
            """
            del name, trainable, add_variable_fn   # unused
            dist = normal_lib.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(3.))
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


        kernel_prior_fn      = nondefault_multivariate_normal_fn

        kernel_posterior_fn  = tfp.layers.default_mean_field_normal_fn( is_singular                     = False, 
                                                                        loc_initializer                 = initializers.random_normal(mean=0.,    stddev=0.1),
                                                                        untransformed_scale_initializer = initializers.random_normal(mean=-20.0, stddev=0.1), 
                                                                        loc_regularizer                 = None, 
                                                                        untransformed_scale_regularizer = None,
                                                                        loc_constraint                  = None, 
                                                                        untransformed_scale_constraint  = None
                                                                        )

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.batch_size * 1.0)


        bias_prior_fn        = None

        bias_posterior_fn    = tfp.layers.default_mean_field_normal_fn( is_singular                     = True, 
                                                                        loc_initializer                 = initializers.random_normal(mean=0.,    stddev=0.1),
                                                                        untransformed_scale_initializer = initializers.random_normal(mean=-20.0, stddev=0.1), 
                                                                        loc_regularizer                 = None, 
                                                                        untransformed_scale_regularizer = None,
                                                                        loc_constraint                  = None, 
                                                                        untransformed_scale_constraint  = None
                                                                        )

        bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.batch_size * 1.0)



        layer     = tfp.layers.DenseFlipout( units                = np.int32(self.n_neurons),
                                             activation           = self.act_func,
                                             kernel_prior_fn      = kernel_prior_fn,
                                             kernel_posterior_fn  = kernel_posterior_fn,
                                             kernel_divergence_fn = kernel_divergence_fn,
                                             bias_prior_fn        = bias_prior_fn,
                                             bias_posterior_fn    = bias_posterior_fn,
                                             bias_divergence_fn   = bias_divergence_fn,
                                             name                 = self.layer_name)

        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer



    # ---------------------------------------------------------------------------------------------------------------------------
    def build_TFP_DenseLocal(self):

        def nondefault_multivariate_normal_fn(dtype, shape, name, trainable, add_variable_fn):
            """Creates multivariate standard `Normal` distribution.
            Args:
            dtype: Type of parameter's event.
            shape: Python `list`-like representing the parameter's event shape.
            name: Python `str` name prepended to any created (or existing)
              `tf.Variable`s.
            trainable: Python `bool` indicating all created `tf.Variable`s should be
              added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
            add_variable_fn: `tf.get_variable`-like `callable` used to create (or
              access existing) `tf.Variable`s.
            Returns:
            Multivariate standard `Normal` distribution.å=
            """
            del name, trainable, add_variable_fn   # unused
            dist = normal_lib.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(5.))
            batch_ndims = tf.size(dist.batch_shape_tensor())
            return independent_lib.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


        kernel_prior_fn      = nondefault_multivariate_normal_fn

        kernel_posterior_fn  = tfp.layers.default_mean_field_normal_fn( is_singular                     = False, 
                                                                        loc_initializer                 = initializers.random_normal(mean=0.,   stddev=0.1),
                                                                        untransformed_scale_initializer = initializers.random_normal(mean=-15.0, stddev=0.1), 
                                                                        loc_regularizer                 = None, 
                                                                        untransformed_scale_regularizer = None,
                                                                        loc_constraint                  = None, 
                                                                        untransformed_scale_constraint  = None
                                                                        )

        kernel_divergence_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.batch_size * 1.0)


        bias_prior_fn        = None

        bias_posterior_fn    = tfp.layers.default_mean_field_normal_fn( is_singular                     = True, 
                                                                        loc_initializer                 = initializers.random_normal(mean=0.,   stddev=0.1),
                                                                        untransformed_scale_initializer = initializers.random_normal(mean=-15.0, stddev=0.1), 
                                                                        loc_regularizer                 = None, 
                                                                        untransformed_scale_regularizer = None,
                                                                        loc_constraint                  = None, 
                                                                        untransformed_scale_constraint  = None
                                                                        )

        bias_divergence_fn   = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (self.batch_size * 1.0)


        layer     = tfp.layers.DenseLocalReparameterization( units                = np.int32(self.n_neurons),
                                                             activation           = self.act_func,
                                                             activity_regularizer = None,
                                                             kernel_prior_fn      = kernel_prior_fn,
                                                             kernel_posterior_fn  = kernel_posterior_fn,
                                                             kernel_divergence_fn = kernel_divergence_fn,
                                                             bias_prior_fn        = bias_prior_fn,
                                                             bias_posterior_fn    = bias_posterior_fn,
                                                             bias_divergence_fn   = bias_divergence_fn,
                                                             name                 = self.layer_name)

        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer


    # ---------------------------------------------------------------------------------------------------------------------------
    def build_TFP_DenseVariational(self):
        # https://colab.research.google.com/github/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb#scrollTo=VwzbWw3_CQ2z

        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            c = np.log(np.expm1(1.))
            return tf.keras.Sequential([tfp.layers.VariableLayer(2 * n, dtype=dtype, name=self.layer_name+'_1'),
                                        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(tfp.distributions.Normal(loc=t[..., :n], 
                                                                                                                                       scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])), 
                                                                                                              reinterpreted_batch_ndims=1)),
                                       ])

        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
        def prior_trainable(kernel_size, bias_size=0, dtype=None):
            n = kernel_size + bias_size
            return tf.keras.Sequential([tfp.layers.VariableLayer(n, dtype=dtype, name=self.layer_name+'_2'),
                                        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(tfp.distributions.Normal(loc=t, scale=3.),
                                                                                                              reinterpreted_batch_ndims=1)),
                                       ])


        layer  = tfp.layers.DenseVariational(units                = np.int32(self.n_neurons),
                                             make_posterior_fn    = posterior_mean_field, 
                                             make_prior_fn        = prior_trainable, 
                                             kl_weight            = 1. / self.batch_size,
                                             kl_use_exact         = False,
                                             use_bias             = True, 
                                             activity_regularizer = None,
                                             activation           = self.act_func,
                                             name                 = self.layer_name)

        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer

