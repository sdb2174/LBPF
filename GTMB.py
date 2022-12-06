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


def GTMB(x):
    conv_func = conv_relu
    t = tf.keras.layers.ZeroPadding2D(padding = (1, 1)) (x) 
    t = conv_func(t, 256, 3, strides=(2,2))
    t = tf.keras.layers.GlobalAveragePooling2D()(t)
    t = tf.keras.layers.Dense(1024,activation='relu')(t)
    t = tf.keras.layers.Dense(512, activation='relu')(t)
    t = tf.keras.layers.Dense(256)(t)
    _t = conv_func(x, 256, 1)
    _t = tf.keras.layers.Multiply()([_t, t])
    _t = conv_func(_t, 128, 1)
    return _t