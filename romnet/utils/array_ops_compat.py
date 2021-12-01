import numpy as np

import tensorflow as tf

from .. import config


def istensorlist(values):
    return any(map(tf.is_tensor, values))


def convert_to_array(value):
    """Convert a list to numpy array or tensorflow tensor."""
    if istensorlist(value):
        return tf.convert_to_tensor(value, dtype=config.real(tf))
    value = np.array(value)
    if value.dtype != config.real(np):
        return value.astype(config.real(np))
    return value


def hstack(tup):
    if not tf.is_tensor(tup[0]) and tup[0] == []:
        tup = list(tup)
        if istensorlist(tup[1:]):
            tup[0] = tf.convert_to_tensor([], dtype=config.real(tf))
        else:
            tup[0] = np.array([], dtype=config.real(np))
    return tf.concat(tup, 0) if tf.is_tensor(tup[0]) else np.hstack(tup)


def roll(a, shift, axis):
    return tf.roll(a, shift, axis) if tf.is_tensor(a) else np.roll(a, shift, axis=axis)


def zero_padding(array, pad_width):
    # SparseTensor
    if isinstance(array, (list, tuple)) and len(array) == 3:
        indices, values, dense_shape = array
        indices = [(i + pad_width[0][0], j + pad_width[1][0]) for i, j in indices]
        dense_shape = (
            dense_shape[0] + sum(pad_width[0]),
            dense_shape[1] + sum(pad_width[1]),
        )
        return indices, values, dense_shape
    if tf.is_tensor(array):
        return tf.pad(array, tf.constant(pad_width))
    return np.pad(array, pad_width)
