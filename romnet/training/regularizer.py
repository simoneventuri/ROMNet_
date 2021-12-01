import tensorflow as tf



@tf.keras.utils.register_keras_serializable(package='Custom', name='l1')
class L1Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l1, x0):
    self.l1 = l1
    self.x0 = x0

  def __call__(self, x):
    return self.l1 * tf.math.reduce_sum(tf.math.abs(x - self.x0))

  def get_config(self):
    return {'l1': float(self.l1)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l2')
class L2Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l2, x0):
    self.l2 = l2
    self.x0 = x0

  def __call__(self, x):
    return self.l2 * tf.math.reduce_sum(tf.math.square(x - self.x0))

  def get_config(self):
    return {'l2': float(self.l2)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1l2')
class L1L2Regularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l1, l2, x0):
    self.l1 = l1
    self.l2 = l2
    self.x0 = x0

  def __call__(self, x):
    Diff = x - self.x0
    return self.l1 * tf.math.reduce_sum(tf.math.abs(Diff)) + self.l2 * tf.math.reduce_sum(tf.math.square(Diff))

  def get_config(self):
    return {'l1l2': float(self.l2)}
