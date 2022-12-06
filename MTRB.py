from ait import adaptive_implicit_trans
import tensorflow as tf

def conv_relu(x, filters, kernel, padding='same', use_bias = True, dilation_rate=1, strides=(1,1)):
    if dilation_rate == 0:
        y = tf.keras.layers.Conv2D(filters,1,padding=padding,use_bias=use_bias,
            activation='relu')(x)
    else:
        y = tf.keras.layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
            dilation_rate=dilation_rate,
            strides=strides,
            activation='relu')(x)
    return y

def conv(x, filters, kernel, padding='same', use_bias=True, dilation_rate=1, strides = (1,1)):
    y = tf.keras.layers.Conv2D(filters,kernel,padding=padding,use_bias=use_bias,
        dilation_rate=dilation_rate, strides=strides)(x)
    return y

def MTRB(x, d_list, enbale=True):
    t = x
    conv_func = conv_relu
    for i in range(len(d_list)):
        _t = conv_func(t, 64, 3, dilation_rate=d_list[i])
        t = tf.keras.layers.Concatenate(axis=-1)([_t, t])
    t = conv(t, 64, 3)
    t = adaptive_implicit_trans()(t)
    t = conv(t, 128, 1)
    t = tf.keras.layers.Rescaling(scale=0.1)(t)  
    t = tf.keras.layers.Add()([x, t])
    return t
