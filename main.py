import os

import numpy as np
import tensorflow as tf
from keras.preprocessing.image_dataset import image_dataset_from_directory
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K

from utility import evaluate

model_path = 'models'
test_path = 'test_set'
training_set = 'training_set'


def load_model():
    model_name = 'GTSRB.h5'
    #K.set_floatx('posit160')
    model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile=False)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


'''def load_test_set(test_path, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE):
    test_set = image_dataset_from_directory(
        test_path,
        label_mode='categorical',
        shuffle=False,
        labels='inferred',
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE)
    test_set.cardinality().numpy()
    return test_set'''


def load_test_set():
    images = training_set
    data = []
    labels = []
    classes = 43
    for i in range(classes):
        if i < 10:
            img_path = os.path.join(images, "0000" + str(i))
        else:
            img_path = os.path.join(images, "000" + str(i))
        for img in os.listdir(img_path):
            if ".csv" in img:
                continue
            im = Image.open(img_path + '/' + img)
            im = im.resize((32, 32))
            im = np.array(im)
            data.append(im)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)

    x = data.astype('float32')
    y = tf.keras.utils.to_categorical(np.array(labels))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y)
    return x_test, y_test


if __name__ == '__main__':
    gtsrb_model = load_model()
    x_test, y_test = load_test_set()
    evaluate(gtsrb_model, x_test, y_test)
