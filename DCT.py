from math import sqrt, pi, cos
import numpy as np
import tensorflow as tf 

class BlockDCT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BlockDCT, self).__init__(**kwargs)

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
        
        # loop through the rows of the kernel
        for row in range(8):
            row_val = 2 * row + 1
            # loop through the columns of the kernel
            for col in range(8):
                col_val = 2 * col + 1
                # calculate the index for this row and column
                index = row*8+col
                for inner_row in range(8):
                    for inner_col in range(8):
                        inner_index = inner_row * 8 + inner_col
                        value = cos(row_val * inner_row * pi / 16) * cos(col_val * inner_col * pi / 16)
                        value = value * r1 if inner_row == 0 else value * r2
                        value = value * r1 if inner_col == 0 else value * r2
                        kernel[0,0,inner_index,index] = value

                     
        self.kernel = tf.keras.backend.variable(value = kernel, dtype = 'float32')

    def call(self, inputs):
        
        self.kernel = self.kernel * self.it_weights
        
        y = tf.keras.backend.conv2d(inputs,
                        self.kernel,
                        padding = 'same',
                        data_format='channels_last')
        
        
        return y

    def compute_output_shape(self, input_shape):
        return input_shape