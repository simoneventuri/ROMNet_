from .__about__ import __version__

from . import architecture
from . import data
from . import model
from . import pinn
from . import training
from . import utils


# TensorFlow ------------------------------------------------------------------
tf_setup = {
    'EPSILON':              1.e-20,
    'DTYPE':                'float64',
    'NUM_THREADS':          0,
    'TF_CPP_MIN_LOG_LEVEL': '3'
}

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_setup['TF_CPP_MIN_LOG_LEVEL']

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx(tf_setup['DTYPE'])
tf.keras.backend.set_epsilon(tf_setup['EPSILON'])
#tf.config.threading.set_intra_op_parallelism_threads(tf_setup['NUM_THREADS'])
# tf.config.threading.set_intra_op_parallelism_threads(tf_setup['NUM_THREADS'])
