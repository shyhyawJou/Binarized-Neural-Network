import tensorflow as tf
import larq as lq



def QConv_MP_BN(out_dim, kernel_size, input_shape=None):
    '''
    Not quantize the input on first block
    '''

    if input_shape is not None:
        block = tf.keras.Sequential([
            lq.layers.QuantConv2D(filters=out_dim,
                                  kernel_size=kernel_size,
                                  #padding="same",
                                  kernel_quantizer='ste_sign',
                                  kernel_constraint='weight_clip',
                                  use_bias=False,
                                  input_shape=input_shape),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.BatchNormalization(scale=False)
        ])
    else:
        block = tf.keras.Sequential([
            lq.layers.QuantConv2D(filters=out_dim,
                                  kernel_size=kernel_size,
                                  #padding="same",
                                  input_quantizer="ste_sign",
                                  kernel_quantizer='ste_sign',
                                  kernel_constraint='weight_clip',
                                  use_bias=False),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.BatchNormalization(scale=False)
        ])

    return block


def get_model(input_shape, n_class):
    """
    Accuracy Efficient Architecture in the paper
    """
    model = tf.keras.Sequential([
        QConv_MP_BN(32, 5, input_shape=input_shape),
        QConv_MP_BN(64, 5, input_shape=None),
        QConv_MP_BN(64, 3, input_shape=None),
        tf.keras.layers.Flatten(),
        lq.layers.QuantDense(1024, 
                             input_quantizer="ste_sign",
                             kernel_quantizer='ste_sign',
                             kernel_constraint='weight_clip',
                             use_bias=False),
        tf.keras.layers.BatchNormalization(scale=False),
        lq.layers.QuantDense(n_class, 
                             input_quantizer="ste_sign",
                             kernel_quantizer='ste_sign',
                             kernel_constraint='weight_clip',
                             use_bias=False),
        tf.keras.layers.Softmax()
    ], name='BNN')

    return model
