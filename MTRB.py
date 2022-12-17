from DCT import BlockDCT
import tensorflow as tf

def MTRB(x, d_list, enbale=True):
    output = x
    
    # We will perfrom many convolutions using a series of different dilation rates, adn then concatenate them all together.
    for i in range(len(d_list)):
        output_conc = tf.keras.layers.Conv2D(64, 3, padding = 'same', use_bias = True, dilation_rate=d_list[i], strides = (1, 1), activation = 'relu')(output)
        output = tf.keras.layers.Concatenate(axis=-1)([output_conc, output])
       
    # intermediate convolution:
    output = tf.keras.layers.Conv2D(64, 3, padding = 'same', use_bias = True, dilation_rate = 1, strides = (1, 1))(output)
    
    # Perfrom DCT and LP operations (D^-1 block in paper):
    output = BlockDCT()(output)
    
    # One more convolution
    output = tf.keras.layers.Conv2D(128, 1, padding = 'same', use_bias = True, dilation_rate = 1, strides = (1, 1))(output)
    
    # Could not get feature scaling to work.
    
    # Perform addition at the end, so that we have a residual block.
    output = tf.keras.layers.Add()([x, output])
    return output
