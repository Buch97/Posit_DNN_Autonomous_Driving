import os

import tensorflow as tf
from keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow import posit160
from tensorflow.keras import backend as K

from utility import evaluate

models_path = 'models'
test_path = 'test_set'


def load_model():
    model_name = 'GTSRB.h5'
    K.set_floatx('posit160')
    gtsrb_model = tf.keras.models.load_model(os.path.join(models_path, model_name))
    model.summary()
    return gtsrb_model


def load_test_set(test_path, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE):
    test_set = image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        shuffle=False,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE)
    return test_set


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    model = load_model()
    test_set = load_test_set()
    # evaluate(model, test_set)
