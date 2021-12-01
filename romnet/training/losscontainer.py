import copy

import tensorflow as tf

from tensorflow.python.keras.engine.compile_utils import *
from tensorflow.python.distribute import distribution_strategy_context as ds_context
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export     import keras_export


from . import loss as losses_mod 


class LossesContainer(Container):
  """A container class for losses passed to `Model.compile`."""

  def __init__(self, losses, loss_weights=None, output_names=None):
    super(LossesContainer, self).__init__(output_names=output_names)

    # Keep user-supplied values untouched for recompiling and serialization.
    self._user_losses = losses
    self._user_loss_weights = loss_weights

    self._losses = losses
    self._loss_weights = loss_weights
    self._per_output_metrics = None  # Per-output losses become metrics.
    self._loss_metric = metrics_mod.Mean(name='loss')  # Total loss.
    self._built = False

  @property
  def metrics(self):
    """Per-output loss metrics."""
    if not self._built:
      return []
    per_output_metrics = [
        metric_obj for metric_obj in nest.flatten(self._per_output_metrics)
        if metric_obj is not None
    ]
    return [self._loss_metric] + per_output_metrics

  def build(self, y_pred):
    """One-time setup of loss objects."""
    super(LossesContainer, self).build(y_pred)

    self._losses = self._maybe_broadcast_to_outputs(y_pred, self._losses)
    self._losses = self._conform_to_outputs(y_pred, self._losses)
    self._losses = nest.map_structure(self._get_loss_object, self._losses)
    self._losses = nest.flatten(self._losses)

    self._loss_weights = self._maybe_broadcast_to_outputs(
        y_pred, self._loss_weights)
    self._loss_weights = self._conform_to_outputs(y_pred, self._loss_weights)
    self._loss_weights = nest.flatten(self._loss_weights)

    self._create_metrics()
    self._built = True


  @property
  def built(self):
    return self._built


  def _create_metrics(self):
    """Creates per-output loss metrics, but only for multi-output Models."""
    if len(self._output_names) == 1:
      self._per_output_metrics = [None]
    else:
      self._per_output_metrics = []
      for loss_obj, output_name in zip(self._losses, self._output_names):
        if loss_obj is None:
          self._per_output_metrics.append(None)
        else:
          self._per_output_metrics.append(
              metrics_mod.Mean(output_name + '_loss'))


  def __call__(self,
               y_true,
               y_pred,
               attention_mask=None,
               sample_weight=None,
               regularization_losses=None):
    """Computes the overall loss.
    Args:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.
      regularization_losses: Additional losses to be added to the total loss.
    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    y_true         = self._conform_to_outputs(y_pred, y_true)
    attention_mask = self._conform_to_outputs(y_pred, attention_mask)
    sample_weight  = self._conform_to_outputs(y_pred, sample_weight)

    if not self._built:
      self.build(y_pred)

    y_pred         = nest.flatten(y_pred)
    y_true         = nest.flatten(y_true)
    attention_mask = nest.flatten(attention_mask)
    sample_weight  = nest.flatten(sample_weight)

    loss_values = []  # Used for gradient calculation.
    loss_metric_values = []  # Used for loss metric calculation.
    batch_dim = None
    zip_args = (y_true, y_pred, attention_mask, sample_weight, self._losses, self._loss_weights,
                self._per_output_metrics)
    for y_t, y_p, a_m, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
      if y_t is None or loss_obj is None:  # Ok to have no loss for an output.
        continue

      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      sw = apply_mask(y_p, sw, get_mask(y_p))
      loss_value = loss_obj(y_t, y_p, attention_mask=a_m, sample_weight=sw)

      loss_metric_value = loss_value
      # Correct for the `Mean` loss metrics counting each replica as a batch.
      if loss_obj.reduction == losses_utils.ReductionV2.SUM:
        loss_metric_value *= ds_context.get_strategy().num_replicas_in_sync

      if batch_dim is None:
        if tf_utils.is_ragged(y_t):
          batch_dim = y_t.nrows()
        else:
          batch_dim = array_ops.shape(y_t)[0]

      if metric_obj is not None:
        metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)

      if loss_weight is not None:
        loss_value *= loss_weight
        loss_metric_value *= loss_weight

      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)

      loss_values.append(loss_value)
      loss_metric_values.append(loss_metric_value)

    if regularization_losses:
      regularization_losses = losses_utils.cast_losses_to_common_dtype(
          regularization_losses)
      reg_loss = math_ops.add_n(regularization_losses)
      loss_metric_values.append(reg_loss)
      loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

    if loss_values:
      loss_metric_values = losses_utils.cast_losses_to_common_dtype(
          loss_metric_values)
      total_loss_metric_value = math_ops.add_n(loss_metric_values)
      self._loss_metric.update_state(
          total_loss_metric_value, sample_weight=batch_dim)

      loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
      total_loss = math_ops.add_n(loss_values)
      return total_loss
    else:
      # Ok for a model to have no compiled loss.
      return array_ops.zeros(shape=())


  def reset_state(self):
    """Resets the state of loss metrics."""
    if not self._built:
      return
    metrics = [self._loss_metric] + nest.flatten(self._per_output_metrics)
    for metric_obj in metrics:
      if metric_obj is not None:
        metric_obj.reset_state()


  def _get_loss_object(self, loss):
    """Returns a `Loss` object.
    Converts the user-supplied loss to a `Loss` object. Also allows
    `SUM_OVER_BATCH_SIZE` reduction to be used for this loss.
    Args:
      loss: A string, function, or `Loss` object.
    Returns:
      A `Loss` object.
    """
    if loss is None:
      return None  # Ok to have no loss for an output.

    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
      loss_name = get_custom_object_name(loss)
      if loss_name is None:
        raise ValueError('Loss should be a callable, found: {}'.format(loss))
      loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    loss._allow_sum_over_batch_size = True  # pylint: disable=protected-access
    return loss

  def _should_broadcast(self, obj):
    return not nest.is_nested(obj)

  def _copy_object(self, obj):
    return obj  # Losses don't need to be copied.
