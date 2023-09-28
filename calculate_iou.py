import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from RetinaNet import get_backbone, RetinaNetLoss, RetinaNet, DecodePredictions, compute_iou, \
    resize_and_pad_image_posit
from RetinaNet_float32 import RetinaNetFloat32, get_backbone_float32
from utility import parse_tfrecord

NUM_CLASSES = 43
model_dir = '/media/matteo/CIRAGO/weights'

# CONFIGURE MODEL
learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
loss_fn = RetinaNetLoss(NUM_CLASSES)

# LOAD MODEL
K.set_floatx('posit160')
input_shape = (None, None, None, 3)

# posit160 model
resnet50_backbone = get_backbone()
model = RetinaNet(NUM_CLASSES, resnet50_backbone)
model.build(input_shape)
model.compile(loss=loss_fn, optimizer=optimizer)

# float32 model
resnet50_backbone_float32 = get_backbone_float32()
old_model = RetinaNetFloat32(NUM_CLASSES, resnet50_backbone_float32)
old_model.build(input_shape)
old_model.compile(loss=loss_fn, optimizer=optimizer)

print("********LOADING MODEL********")
latest_checkpoint = tf.train.latest_checkpoint(model_dir)
old_model.load_weights(latest_checkpoint)
model.set_weights(old_model.get_weights())
print("********MODEL LOADED********")

model.summary()

image = tf.keras.Input(shape=[None, None, 3], name="image", dtype=K.floatx())
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.4)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

print("********LOAD TEST SET********")
eval_ds = tf.data.TFRecordDataset('dataset/test.record', num_parallel_reads=tf.data.experimental.AUTOTUNE)
eval_ds = eval_ds.map(parse_tfrecord)
print("********TEST SET LOADED********")

width = 1360
height = 800

print("********START INFERENCE********")

i = 0
iou_list = []

print('Mode: ' + K.floatx())


def prepare_image(img):
    img, _, r = resize_and_pad_image_posit(img, jitter=None)
    img = tf.keras.applications.resnet.preprocess_input(img)
    img = tf.cast(img, dtype=K.floatx())
    return tf.expand_dims(img, axis=0), r


for sample in eval_ds:
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image, verbose=1)
    num_detections = detections.valid_detections[0]

    pred = (detections.nmsed_boxes[0][:num_detections] / ratio)
    pred = tf.cast(pred, dtype=K.floatx())

    normalized_pred = pred / tf.constant([width, height, width, height], dtype=K.floatx())
    iou = compute_iou(normalized_pred, tf.cast(sample["objects"]["bbox"], dtype=K.floatx()))
    iou_list.extend(iou)

    i = i + 1
    print('Image ' + str(i))

means = [tf.reduce_mean(arr).numpy() for arr in iou_list]
global_mean = np.mean(means)

print("Average IOU:", global_mean)
