import tensorflow as tf
from keras.layers import LayerNormalization, BatchNormalization
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


def my_clone_function(layer):
    if isinstance(layer, Model):
        return clone_model(layer, clone_function=my_clone_function)
    if isinstance(layer, BatchNormalization) or isinstance(layer, LayerNormalization):
        config = layer.get_config()
        return layer.__class__.from_config(config)
    if isinstance(layer, Layer):
        config = layer.get_config()
        config['dtype'] = K.floatx()
        return layer.__class__.from_config(config)


def clone_old_model(original_model):
    model = clone_model(original_model, clone_function=my_clone_function)
    return model


'''def create_model_resnet(weights):
    input_shape = (32, 32, 3)

    conv_base = tf.keras.applications.resnet_v2.ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape)
    conv_base.trainable = False

    pretrained_weights = [w.astype(K.floatx()) for w in conv_base.get_weights()]

    inputs = tf.keras.Input(shape=input_shape, dtype=K.floatx())
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = conv_base(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='my_glo_avg_pool')(x)

    outputs = tf.keras.layers.Dense(43, activation="softmax", name='predictions', dtype=K.floatx())(x)
    model = tf.keras.Model(inputs, outputs)
    for layer in model.layers:
        layer._dtype = K.floatx()
    for layer in model.layers[3].layers:
        layer._dtype = K.floatx()
    model.set_weights(weights)
    model.set_weights(pretrained_weights)
    return model'''
