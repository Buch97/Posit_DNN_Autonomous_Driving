import os
import random
import sys

import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
import cv2
from PIL import Image

model_path = 'models'
datasets_folder = 'dataset'


def data_augmentation(original_image):
    if random.randint(0, 1):  # decide whether to flip the image or not
        horizontal = random.randint(0, 1)
        if horizontal:
            new_image = tf.image.flip_left_right(original_image)
        else:
            new_image = tf.image.flip_up_down(original_image)
    else:  # rotate the image of a random degree (between 90° and 270°)
        k = random.randint(1, 3)
        new_image = tf.image.rot90(original_image, k)

    return np.asarray(new_image)


def evaluate(model, x_test, y_test):
    y_score = model.predict(x_test)
    y_pred = np.argmax(y_score, axis=1)
    y_true = np.argmax(y_test, axis=1)
    correct_predictions = np.sum(np.equal(y_pred, y_true))
    accuracy = correct_predictions / len(y_true)
    print("accuracy: " + str(accuracy))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.3f}, test loss: {test_loss:.3f}")

    print("Classification report: ")
    # print(type(y_true))
    # print(type(y_pred))
    # print(y_pred.argmax(axis=1))
    print(metrics.classification_report(y_true, y_pred, digits=4))
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

    '''# ROC curve
    fpr, tpr, th = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()'''


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
            data.append(im)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)

    x = data.astype('float32')
    y = tf.keras.utils.to_categorical(np.array(labels))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y)
    return x_test, y_test


def load_test_set_gtsdb(dataset):
    print("DATASET: " + dataset)
    images = os.path.join(datasets_folder, dataset)

    for i in os.listdir(images):
        img_path = os.path.join(images, i)
        for img in os.listdir(img_path):
            im = cv2.imread(img_path + '/' + img)
            if int(i) == 31 or int(i) == 37 or int(i) == 27 or int(i) == 19 or int(i) == 00:
                new_img = data_augmentation(im)
                path = os.path.join(img_path, "aug_" + img)
                if not cv2.imwrite(path, new_img):
                    raise Exception("Could not write image")

    data = []
    labels = []

    for i in os.listdir(images):
        img_path = os.path.join(images, i)
        for img in os.listdir(img_path):
            im = Image.open(img_path + '/' + img)
            im = im.resize((32, 32))
            im = np.array(im)
            data.append(im)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)

    x = data.astype('float32')
    y = tf.keras.utils.to_categorical(np.array(labels))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y)
    return x_test, y_test


def load_model(model_name, exec_mode):
    if exec_mode != 'float32' and exec_mode != 'posit':
        print("Error exec mode")
        sys.exit()
    elif exec_mode == 'posit':
        K.set_floatx('posit160')
        print('Posit mode')
    elif exec_mode == 'float32':
        print("Float32 mode")

    model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile=False)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
