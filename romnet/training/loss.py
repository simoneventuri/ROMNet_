import tensorflow as tf
import numpy      as np

import abc
import functools

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.python.keras       import backend as K


# Custom loss functions
###########################################################################
@tf.keras.utils.register_keras_serializable(package='ROMNet', name='MAE')
def mae(axis):
    def mean_absolute_error(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        err    = tf.math.abs(y_pred - y_true)
        return tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_absolute_error


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='MALPE')
def malpe(axis):
    def mean_absolute_log_percentage_error(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(K.maximum(y_pred, K.epsilon()))
        y_true = tf.math.log(K.maximum(y_true, K.epsilon()))
        err    = tf.math.abs((y_true - y_pred) / y_true)
        return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_absolute_log_percentage_error


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='MAPE')
def mape(axis):
    def mean_absolute_percentage_error(y_true, y_pred, attention_mask=None):
        #n_points = tf.cast(tf.shape(y_true)[0], tf.float64)
        y_pred   = tf.convert_to_tensor(y_pred)
        y_true   = tf.cast(y_true, y_pred.dtype)
        err      = tf.math.abs( (y_true - y_pred) / K.maximum(tf.math.abs(y_true), K.epsilon()) )
        if attention_mask is not None:
            attention_mask = tf.convert_to_tensor(attention_mask)
            err           *= attention_mask**2
        #return 100. * tf.reduce_sum(tf.reduce_mean(err, axis=axis))
        return 100. * K.mean(err, axis=-1) #/ (n_points)

    return mean_absolute_percentage_error


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='MSE')
def mse(axis):
    def mean_squared_error(y_true, y_pred, attention_mask=None):
        #n_points = tf.cast(tf.shape(y_true)[0], tf.float64)
        y_pred   = tf.convert_to_tensor(y_pred)
        y_true   = tf.cast(y_true, y_pred.dtype)
        err      = tf.math.squared_difference(y_pred, y_true) 
        if attention_mask is not None:
            attention_mask = tf.convert_to_tensor(attention_mask)
            err           *= attention_mask**2
        # return K.mean(err, axis=-1)                          # TF
        # return tf.reduce_sum(tf.reduce_mean(err, axis=axis)) # PRODE
        return K.mean(err, axis=-1) #/ (n_points)

    return mean_squared_error


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='MSLE')
def msle(axis):
    def mean_squared_logarithmic_error(y_true, y_pred):
        # Warning: see MeanSquaredError above.
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.math.log(K.maximum(y_pred, K.epsilon()))
        y_true = tf.math.log(K.maximum(y_true, K.epsilon()))
        err    = tf.math.squared_difference(y_pred, y_true)
        return tf.reduce_sum(tf.reduce_mean(err, axis=axis))
    return mean_squared_logarithmic_error


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='ZERO')
def zero(*args):
    return tf.constant(0., dtype=K.floatx())


@tf.keras.utils.register_keras_serializable(package='ROMNet', name='NLL')
def nll(axis): 
    def negative_log_likelihood(y, distr, attention_mask=None):
        #y_true   = tf.cast(y_true, y_pred.dtype)
        return K.sum( -distr.log_prob(y), axis=-1)
    return negative_log_likelihood




def get_loss(name, axis=-1):

    if isinstance(name, (list, tuple)):
        return list(map(get_loss, name))
    
    if isinstance(name, dict):
        name, axis = name.values()

    LF = name.lower()
    if (LF == 'binary_crossentropy'):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy")

    elif (LF == 'categorical_crossentropy'):
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0, reduction="auto", name="categorical_crossentropy",)

    elif (LF == 'sparse_categorical_crossentropy'):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy")

    elif (LF == 'poisson'):
        return tf.keras.losses.Poisson(reduction="auto", name="poisson")

    elif (LF == 'binary_crossenkl_divergencetropy'):
        return tf.keras.losses.KLDivergence(reduction="auto", name="kl_divergence")

    elif (LF == 'mean_squared_error'):
        return tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

    elif (LF == 'mean_absolute_error'):
        return tf.keras.losses.MeanAbsoluteError(reduction="auto", name="mean_absolute_error")

    elif (LF == 'mean_absolute_percentage_error'):
        return tf.keras.losses.MeanAbsolutePercentageError(reduction="auto", name="mean_absolute_percentage_error")

    elif (LF == 'mean_squared_logarithmic_error'):
        return tf.keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")

    elif (LF == 'cosine_similarity'):
        return tf.keras.losses.CosineSimilarity(axis=-1, reduction="auto", name="cosine_similarity")

    elif (LF == 'huber_loss'):
        return tf.keras.losses.Huber(delta=1.0, reduction="auto", name="huber_loss")

    elif (LF == 'log_cosh'):
        return tf.keras.losses.LogCosh(reduction="auto", name="log_cosh")

    elif (LF == 'hinge'):
        return tf.keras.losses.Hinge(reduction="auto", name="hinge")

    elif (LF == 'squared_hinge'):
        return tf.keras.losses.SquaredHinge(reduction="auto", name="squared_hinge")

    elif (LF == 'categorical_hinge'):
        return tf.keras.losses.CategoricalHinge(reduction="auto", name="categorical_hinge")

    elif (LF == 'mae'):
        return mae(axis)

    elif (LF == 'malpe'):
        return malpe(axis)

    elif (LF == 'mape'):
        return mape(axis)

    elif (LF == 'mse'):
        return mse(axis)

    elif (LF == 'msle'):
        return mae(axis)

    elif (LF == 'zero'):
        return zero

    elif (LF == 'nll'):
        return nll(axis)

    elif (LF == None):
        return None
    else:
        raise ValueError("Unrecognized Loss Function!   InputData.LossFunction = ", InputData.LossFunction)




@keras_export('keras.losses.Loss')
class Loss:
  """Loss base class.
  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
  Example subclass implementation:
  ```python
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```
  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.
  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.
  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    """Initializes `Loss` class.
    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
    """
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
    # CannedEstimators.
    self._allow_sum_over_batch_size = False
    self._set_name_scope()

  def _set_name_scope(self):
    """Creates a valid `name_scope` name."""
    if self.name is None:
      self._name_scope = self.__class__.__name__
    elif self.name == '<lambda>':
      self._name_scope = 'lambda'
    else:
      # E.g. '_my_loss' => 'my_loss'
      self._name_scope = self.name.strip('_')

  def __call__(self, y_true, y_pred, attention_mask=None, sample_weight=None):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      sample_weight: Optional `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `y_pred` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
        because all loss functions reduce by 1 dimension, usually axis=-1.)
    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, attention_mask, sample_weight)
    with backend.name_scope(self._name_scope), graph_ctx:
      if context.executing_eagerly():
        call_fn = self.call
      else:
        call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = call_fn(y_true, y_pred, attention_mask=attention_mask)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Loss` from its config (output of `get_config()`).
    Args:
        config: Output of `get_config()`.
    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    """Returns the config dictionary for a `Loss` instance."""
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if (not self._allow_sum_over_batch_size and
        distribution_strategy_context.has_strategy() and
        (self.reduction == losses_utils.ReductionV2.AUTO or
         self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction


class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    """Initializes `LossFunctionWrapper` class.
    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super().__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred, attention_mask=None):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tf_type(y_pred) and tensor_util.is_tf_type(y_true):
      y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, attention_mask=attention_mask, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))



@keras_export('keras.losses.get')
def get(identifier):
  """Retrieves a Keras loss as a `function`/`Loss` class instance.
  The `identifier` may be the string name of a loss function or `Loss` class.
  >>> loss = tf.keras.losses.get("categorical_crossentropy")
  >>> type(loss)
  <class 'function'>
  >>> loss = tf.keras.losses.get("CategoricalCrossentropy")
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>
  You can also specify `config` of the loss to this function by passing dict
  containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Loss` class
  >>> identifier = {"class_name": "CategoricalCrossentropy",
  ...               "config": {"from_logits": True}}
  >>> loss = tf.keras.losses.get(identifier)
  >>> type(loss)
  <class '...keras.losses.CategoricalCrossentropy'>
  Args:
    identifier: A loss identifier. One of None or string name of a loss
      function/class or loss configuration dictionary or a loss function or a
      loss class instance.
  Returns:
    A Keras loss as a `function`/ `Loss` class instance.
  Raises:
    ValueError: If `identifier` cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  if callable(identifier):
    return identifier
  raise ValueError(
      f'Could not interpret loss function identifier: {identifier}')
