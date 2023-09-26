import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from RetinaNet import LabelEncoder, get_backbone, RetinaNetLoss, RetinaNet, DecodePredictions, compute_iou, \
    resize_and_pad_image
from RetinaNet_float32 import RetinaNetFloat32, get_backbone_float32
from utility import parse_tfrecord

NUM_CLASSES = 43
model_dir = '/media/matteo/CIRAGO/weights'
class_ids = {
    'speed_limit_20': 1,
    'speed_limit_30': 2,
    'speed_limit_50': 3,
    'speed_limit_60': 4,
    'speed_limit_70': 5,
    'speed_limit_80': 6,
    'restriction_ends_80': 7,
    'speed_limit_100': 8,
    'speed_limit_120': 9,
    'no_overtaking': 10,
    'no_overtaking_trucks': 11,
    'priority_at_next_intersection': 12,
    'priority_road': 13,
    'give_way': 14,
    'stop': 15,
    'no_traffic_both_ways': 16,
    'no_trucks': 17,
    'no_entry': 18,
    'danger': 19,
    'bend_left': 20,
    'bend_right': 21,
    'bend': 22,
    'uneven_road': 23,
    'slippery_road': 24,
    'road_narrows': 25,
    'construction': 26,
    'traffic_signal': 27,
    'pedestrian_crossing': 28,
    'school_crossing': 29,
    'cycles_crossing': 30,
    'snow': 31,
    'animals': 32,
    'restriction_ends': 33,
    'go_right': 34,
    'go_left': 35,
    'go_straight': 36,
    'go_right_or_straight': 37,
    'go_left_or_straight': 38,
    'keep_right': 39,
    'keep_left': 40,
    'roundabout': 41,
    'restriction_ends_overtaking': 42,
    'restriction_ends_overtaking_trucks': 43
}
keys_array = list(class_ids.keys())
class_mapping = dict(zip(range(1, len(class_ids) + 1), class_ids))
iou_list = []

# CONFIGURE MODEL
label_encoder = LabelEncoder()

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

# LOAD MODEL
K.set_floatx('posit160')

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(NUM_CLASSES)
model = RetinaNet(NUM_CLASSES, resnet50_backbone)
old_model = RetinaNetFloat32(NUM_CLASSES, get_backbone_float32())

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)
latest_checkpoint = tf.train.latest_checkpoint(model_dir)

print("********LOADING MODEL********")
old_model.load_weights(latest_checkpoint)
model.set_weights(old_model.get_weights())
print("********MODEL LOADED********")

image = tf.keras.Input(shape=[None, None, 3], name="image", dtype=K.floatx())
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.4)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

print("********LOAD TEST SET********")
eval_ds = tf.data.TFRecordDataset('dataset/test.record', num_parallel_reads=tf.data.experimental.AUTOTUNE)
eval_ds = eval_ds.map(parse_tfrecord)
print("********TEST SET LOADED********")

width = 1300
height = 800

print("********START INFERENCE********")

i = 0


def prepare_image(img):
    img, _, r = resize_and_pad_image(img, jitter=None)
    img = tf.keras.applications.resnet.preprocess_input(img)
    img = tf.cast(img, dtype=K.floatx())
    return tf.expand_dims(img, axis=0), r


for sample in eval_ds:
    input_image, ratio = prepare_image(sample["image"])
    detections = inference_model.predict(input_image, verbose=1)
    i = i + 1
    print('Image ' + str(i))
    num_detections = detections.valid_detections[0]

    pred = (detections.nmsed_boxes[0][:num_detections] / ratio)
    pred = tf.cast(pred, dtype=K.floatx())
    normalized_pred = pred / tf.constant([width, height, width, height], dtype=K.floatx())
    iou = compute_iou(normalized_pred, tf.cast(sample["objects"]["bbox"], dtype=K.floatx()))
    iou_list.extend(iou)

means = [tf.reduce_mean(arr).numpy() for arr in iou_list]
print(means)
global_mean = np.mean(means)

print("Average IOU:", global_mean)
