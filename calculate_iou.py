import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from RetinaNet import LabelEncoder, get_backbone, RetinaNetLoss, RetinaNet, DecodePredictions, resize_and_pad_image, \
    compute_iou

BATCH_SIZE = 8
NUM_CLASSES = 43
EPOCHS = 50
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
iou_list = []

keys_array = list(class_ids.keys())
class_mapping = dict(zip(range(1, len(class_ids) + 1), class_ids))


def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_image(example['image/encoded'])
    image.set_shape((None, None, 3))

    xmin = example['image/object/bbox/xmin'].values
    ymin = example['image/object/bbox/ymin'].values
    xmax = example['image/object/bbox/xmax'].values
    ymax = example['image/object/bbox/ymax'].values
    class_id = example['image/object/class/label'].values

    bounding_box = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    area = (xmax - xmin) * (ymax - ymin)

    objects = {
        'area': tf.cast(area, dtype=tf.float32),
        'bbox': tf.cast(bounding_box, dtype=tf.float32),
        'id': tf.cast(class_id, dtype=tf.float32),
        'is_crowd': False,
        'label': tf.cast(class_id, dtype=tf.float32),
    }

    output_dict = {
        'image': image,
        'image/filename': example['image/filename'],
        'image/source_id': example['image/source_id'],
        'objects': objects,
    }

    return output_dict


# CONFIGURE MODEL
label_encoder = LabelEncoder()

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

# LOAD MODEL
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(NUM_CLASSES)
model = RetinaNet(NUM_CLASSES, resnet50_backbone)

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
K.set_floatx('float16')
model.load_weights(latest_checkpoint)
image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.4)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)


def prepare_image(image):
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    image = tf.cast(image, dtype=tf.float16)
    return tf.expand_dims(image, axis=0), ratio


eval_ds = tf.data.TFRecordDataset('dataset/test.record')
eval_ds = eval_ds.map(parse_tfrecord)
width = 1300
height = 800

for sample in eval_ds:
    input_image, ratio = prepare_image(sample["image"])
    detections = inference_model.predict(input_image, verbose=0)
    num_detections = detections.valid_detections[0]

    pred = (detections.nmsed_boxes[0][:num_detections] / ratio)
    pred = tf.cast(pred, dtype=tf.float16)
    normalized_pred = pred / tf.constant([width, height, width, height], dtype=tf.float16)
    iou = compute_iou(normalized_pred, tf.cast(sample["objects"]["bbox"], dtype=tf.float16))
    iou_list.extend(iou)

means = [tf.reduce_mean(arr).numpy() for arr in iou_list]
print(means)
global_mean = np.mean(means)

print("Average IOU:", global_mean)
