import tensorflow as tf
from keras.layers import LayerNormalization


def create_model_dense():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[32, 32, 3], dtype=K.floatx()))
    model.add(LayerNormalization())
    model.add(tf.keras.layers.Dense(300, activation="relu", dtype=K.floatx()))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(300, activation="relu", dtype=K.floatx()))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(300, activation="relu", dtype=K.floatx()))
    model.add(tf.keras.layers.Dense(43, activation="softmax", dtype=K.floatx()))
    return model


def create_model_conv():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), dtype=K.floatx()))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), dtype=K.floatx()))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.Dense(43, activation='softmax', dtype=K.floatx()))
    return model


def create_model_conv_deeper():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', dtype=K.floatx()))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', dtype=K.floatx()))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', dtype=K.floatx()))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same', dtype=K.floatx()))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.Dense(128, activation='relu', dtype=K.floatx()))
    model.add(tf.keras.layers.Dense(43, activation='softmax', dtype=K.floatx()))
    return model


def create_model_resnet():
    input_shape = (32, 32, 3)

    conv_base = tf.keras.applications.resnet_v2.ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape)
    conv_base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = conv_base(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='my_glo_avg_pool')(x)

    outputs = tf.keras.layers.Dense(43, activation="softmax", name='predictions', dtype=K.floatx())(x)
    model = tf.keras.Model(inputs, outputs)
    return model