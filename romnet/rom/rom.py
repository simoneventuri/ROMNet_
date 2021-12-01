import numpy                                  as np
import tensorflow                             as tf



#=======================================================================================================================================
class PCALayer(tf.keras.layers.Layer):

    def __init__(self, A0, C0, D0, trainable_flg=False, LayerName='PCA'):
        super(PCALayer, self).__init__(name=LayerName)
        self.A0            = A0
        self.C0            = C0
        self.D0            = D0
        self.trainable_flg = trainable_flg

    def build(self, input_shape):
        AIni      = tf.keras.initializers.constant(value=self.A0)
        CIni      = tf.keras.initializers.constant(value=self.C0)
        DIni      = tf.keras.initializers.constant(value=self.D0)
        self.A    = self.add_weight('A',
                                    shape       = self.A0.shape,
                                    initializer = AIni,
                                    trainable   = self.trainable_flg)
        self.C    = self.add_weight('C',
                                    shape       = self.C0.shape,
                                    initializer = CIni,
                                    trainable   = self.trainable_flg)
        self.D    = self.add_weight('D',
                                    shape       = self.D0.shape,
                                    initializer = DIni,
                                    trainable   = self.trainable_flg)

    def call(self, x):
        return tf.matmul( (x - self.C) / self.D, self.A, transpose_b=True)

#=======================================================================================================================================



#=======================================================================================================================================
class AntiPCALayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, trainable_flg=False, LayerName='AntiPCA'):
        super(AntiPCALayer, self).__init__(name=LayerName)
        self.A0            = A0
        self.C0            = C0
        self.D0            = D0
        self.trainable_flg = trainable_flg

    def build(self, input_shape):
        AIni      = tf.keras.initializers.constant(value=self.A0)
        CIni      = tf.keras.initializers.constant(value=self.C0)
        DIni      = tf.keras.initializers.constant(value=self.D0)
        self.A    = self.add_weight('A',
                                    shape       = self.A0.shape,
                                    initializer = AIni,
                                    trainable   = self.trainable_flg)
        self.C    = self.add_weight('C',
                                    shape       = self.C0.shape,
                                    initializer = CIni,
                                    trainable   = self.trainable_flg)
        self.D    = self.add_weight('D',
                                    shape       = self.D0.shape,
                                    initializer = DIni,
                                    trainable   = self.trainable_flg)

    def call(self, x):
        return tf.matmul(x, self.A ) * self.D + self.C

#=======================================================================================================================================
