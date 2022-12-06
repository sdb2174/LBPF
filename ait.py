from math import sqrt, pi, cos
import numpy as np
import tensorflow as tf 

class adaptive_implicit_trans(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(adaptive_implicit_trans, self).__init__(**kwargs)

    def build(self, input_shape):
        conv_shape = (1,1,64,64)
        self.it_weights = self.add_weight(
            shape = (1,1,64,1),
            initializer = tf.keras.initializers.get('ones'),
            constraint = tf.keras.constraints.NonNeg(),
            name = 'ait_conv')
        kernel = np.zeros(conv_shape)
        r1 = sqrt(1.0/8)
        r2 = sqrt(2.0/8)
        for i in range(8):
            _u = 2*i+1
            for j in range(8):
                _v = 2*j+1
                index = i*8+j
                for u in range(8):
                    for v in range(8):
                        index2 = u*8+v
                        t = cos(_u*u*pi/16)*cos(_v*v*pi/16)
                        t = t*r1 if u==0 else t*r2
                        t = t*r1 if v==0 else t*r2
                        kernel[0,0,index2,index] = t
        self.kernel = tf.keras.backend.variable(value = kernel, dtype = 'float32')

    def call(self, inputs):
        #it_weights = k.softmax(self.it_weights)
        #self.kernel = self.kernel*it_weights
        self.kernel = self.kernel*self.it_weights
        y = tf.keras.backend.conv2d(inputs,
                        self.kernel,
                        padding = 'same',
                        data_format='channels_last')
        return y

    def compute_output_shape(self, input_shape):
        return input_shape