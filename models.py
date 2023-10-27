import tensorflow as tf
from keras.layers import LayerNormalization, BatchNormalization
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model


def my_clone_function(layer):
    if isinstance(layer, Model):
        return clone_model(layer, clone_function=my_clone_function)
    if isinstance(layer, BatchNormalization) or isinstance(layer, LayerNormalization):
        config = layer.get_config()
        return layer.__class__.from_config(config)
    if isinstance(layer, Layer):
        config = layer.get_config()
        config['dtype'] = tf.float64
        return layer.__class__.from_config(config)


def clone_old_model(original_model):
    model = clone_model(original_model, clone_function=my_clone_function)
    return model
