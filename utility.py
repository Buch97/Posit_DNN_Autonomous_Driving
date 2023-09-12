import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K

from models import create_model_conv_deeper, create_model_dense, create_model_resnet, create_model_conv

model_path = 'models'
datasets_folder = 'dataset'
random_seed = 42


def evaluate(model, x_test, y_test):
    y_score = model.predict(x_test)
    print(type(y_score[0][0]))
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)
    correct_predictions = np.sum(np.equal(y_pred, y_true))
    accuracy = correct_predictions / len(y_true)
    print("accuracy: " + str(accuracy))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}")

    print("Classification report: ")
    print(metrics.classification_report(y_true, y_pred, digits=4))
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)


def load_test_set_gtsrb(dataset):
    print("DATASET: " + dataset)
    images = os.path.join(datasets_folder, dataset)

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
            if K.floatx() == 'posit160':
                im = tf.convert_to_tensor(im, dtype=tf.posit160)
            elif K.floatx() == 'float32':
                im = tf.convert_to_tensor(im, dtype=tf.float32)
            elif K.floatx() == 'float64':
                im = tf.convert_to_tensor(im, dtype=tf.float64)
            elif K.floatx() == 'float16':
                im = tf.convert_to_tensor(im, dtype=tf.float16)
            data.append(im)
            labels.append(i)

    x = np.array(data)
    labels = np.array(labels)

    y = tf.keras.utils.to_categorical(np.array(labels))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y,
                                                        random_state=random_seed)
    return x_test, y_test


def load_class_model(model_name, exec_mode):
    if exec_mode != 'float32' and exec_mode != 'posit' and exec_mode != 'float16' and exec_mode != 'float64':
        print("Error exec mode")
        sys.exit()
    elif exec_mode == 'posit':
        K.set_floatx('posit160')
        print('Posit mode')
    elif exec_mode == 'float32':
        print("Float32 mode")
    elif exec_mode == 'float16':
        K.set_floatx('float16')
        print("Float16 mode")
    elif exec_mode == 'float64':
        K.set_floatx('float64')
        print("Float64 mode")

    input_shape = (32, 32, 3)
    old_model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile=False)
    model = create_model_conv_deeper()
    model.build(input_shape)
    model.set_weights(old_model.get_weights())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

