import tensorflow as tf

from models.RetinaNet import get_backbone, RetinaNetLoss, RetinaNet


def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
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
    bbox = {
        "classes": tf.cast(class_id, dtype=tf.float32),
        "boxes": tf.cast(bounding_box, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bbox}


'''def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )'''


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_iou(my_model, dataset):
    iou_list = []
    for images, y_true in dataset:
        y_pred = my_model.predict(images, verbose=0)
        # y_pred = bounding_box.to_ragged(y_pred)
        for true_boxes, pred_boxes in zip(y_true['boxes'], y_pred['boxes']):
            true_boxes = true_boxes[true_boxes[:, 0] > -1]
            if len(pred_boxes > 0):
                for boxT, boxP in zip(true_boxes, pred_boxes):
                    iou = bb_intersection_over_union(boxT, boxP)
                    iou_list.append(iou)
    return iou_list


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

if __name__ == '__main__':
    keys_array = list(class_ids.keys())
    class_mapping = dict(zip(range(1, len(class_ids) + 1), class_ids))

    BATCH_SIZE = 2
    NUM_CLASSES = 43
    EPOCHS = 30

    eval_ds = tf.data.TFRecordDataset('/media/matteo/CIRAGO/keras-cv-models')
    eval_ds = eval_ds.map(parse_tfrecord)

    # eval_ds = eval_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    # inference_resizing = tf.keras.layers.Resizing(640, 640)
    # eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

    base_lr = 0.005
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
    )

    # eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(NUM_CLASSES)
    model = RetinaNet(NUM_CLASSES, resnet50_backbone)

    model.built = True
    model.summary()
    model.load_weights('models/keras_cv.h5')
    # model = tf.keras.models.load_model('models/keras_cv.h5', custom_objects={'RetinaNet': RetinaNet})

    ious = calculate_iou(model, dataset=eval_ds)
    mean = tf.math.reduce_mean(ious)
    print(f"IOU: {mean.numpy():.4f}")
