import tensorflow as tf
from tensorflow.python.ops.distributions.beta import Beta
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def augment(images, labels):
    batch_size = images.shape[0]

    labels = tf.expand_dims(labels, -1)
    labels = tf.cast(labels, tf.float32)
    # Crop
    interval = 0.3
    midpoint = 0.1
    boxes = tf.transpose(
        tf.stack([
            tf.random.uniform([batch_size]) * interval - midpoint,
            tf.random.uniform([batch_size]) * interval - midpoint,
            1 - tf.random.uniform([batch_size]) * interval - midpoint,
            1 - tf.random.uniform([batch_size]) * interval - midpoint
        ]),
        [1, 0]
    )
    box_indices = tf.constant(list(range(batch_size)))
    crop_size = FLAGS.resolution
    images = tf.image.crop_and_resize(
        images,
        boxes,
        box_indices,
        crop_size,
        method='bilinear'
    )
    labels = tf.image.crop_and_resize(
        labels,
        boxes,
        box_indices,
        crop_size,
        method='nearest',
        extrapolation_value=255
    )

    # Flip
    flip_mask = tf.less(tf.random.uniform([batch_size]), 0.5)
    images = tf.stack([tf.image.flip_left_right(images[i]) if flip_mask[i] else images[i] for i in range(batch_size)])
    labels = tf.stack([tf.image.flip_left_right(labels[i]) if flip_mask[i] else labels[i] for i in range(batch_size)])
    labels = tf.squeeze(labels, -1)
    labels = tf.cast(labels, tf.int32)

    return images, labels


def augment_image(images, K=1):
    images = tf.tile(images, [K, 1, 1, 1])
    batch_size = images.shape[0]

    # Crop
    interval = 0.3
    midpoint = 0.1
    boxes = tf.transpose(
        tf.stack([
            tf.random.uniform([batch_size]) * interval - midpoint,
            tf.random.uniform([batch_size]) * interval - midpoint,
            1 - tf.random.uniform([batch_size]) * interval - midpoint,
            1 - tf.random.uniform([batch_size]) * interval - midpoint
        ]),
        [1, 0]
    )
    box_indices = tf.constant(list(range(batch_size)))
    crop_size = FLAGS.resolution
    images = tf.image.crop_and_resize(
        images,
        boxes,
        box_indices,
        crop_size,
        method='bilinear'
    )

    # Flip
    flip_mask = tf.less(tf.random.uniform([batch_size]), 0.5)
    images = tf.stack([tf.image.flip_left_right(images[i]) if flip_mask[i] else images[i] for i in range(batch_size)])

    return images, boxes, flip_mask


def augment_labels(labels, boxes, flip_mask):
    batch_size = labels.shape[0]

    # Crop
    box_indices = tf.constant(list(range(batch_size)))
    crop_size = FLAGS.resolution
    labels = tf.image.crop_and_resize(
        labels,
        boxes,
        box_indices,
        crop_size,
        method='nearest',
        extrapolation_value=0
    )

    # Flip
    labels = tf.stack([tf.image.flip_left_right(labels[i]) if flip_mask[i] else labels[i] for i in range(batch_size)])

    return labels


def reverse_augment_labels(labels, boxes, flip_mask):
    batch_size = labels.shape[0]

    # Flip
    labels = tf.stack([tf.image.flip_left_right(labels[i]) if flip_mask[i] else labels[i] for i in range(batch_size)])

    # Crop
    box_indices = tf.constant(list(range(batch_size)))
    crop_size = FLAGS.resolution
    xl, yl, xr, yr = tf.unstack(boxes, axis=1)
    xr_o = 1 - xr
    yr_o = 1 - yr
    inverse_boxes = tf.stack(
        [
            -xl / (xr - xl),
            -yl / (yr - yl),
            1 + xr_o / (xr - xl),
            1 + yr_o / (yr - yl)
        ],
        axis=-1
    )
    labels = tf.image.crop_and_resize(
        labels,
        inverse_boxes,
        box_indices,
        crop_size,
        method='nearest',
        extrapolation_value=0
    )

    return labels


def resize(images, labels):
    labels = tf.expand_dims(labels, -1)
    labels = tf.cast(labels, tf.float32)
    size = FLAGS.resolution
    images = tf.image.resize(
        images,
        size,
        method=tf.image.ResizeMethod.BILINEAR
    )

    labels = tf.image.resize(
        labels,
        size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    labels = tf.squeeze(labels, -1)
    labels = tf.cast(labels, tf.int32)

    return images, labels


def sharpen(p, T=0):
    p_t = tf.pow(p, 1 / T)
    zero_mask = tf.reduce_sum(p_t, axis=-1, keepdims=True) > 0
    return tf.where(
        condition=zero_mask,
        x=p_t / tf.reduce_sum(p_t, axis=-1, keepdims=True),
        y=tf.zeros_like(p_t)
    )


def average_preds(preds, K, all=False):
    if not all:
        return tf.reduce_sum(
            tf.reshape(preds, [FLAGS.batch_size, K, *preds.shape[1:]]),
            axis=1
        ) * tf.reduce_sum(tf.reshape(
            tf.cast(
                preds > 0,
                tf.float32
            ),
            [FLAGS.batch_size, K, *preds.shape[1:]]
        ), axis=1)
    else:
        return tf.reduce_sum(
            tf.reshape(preds, [FLAGS.batch_size, K, *preds.shape[1:]]),
            axis=1
        ) * tf.reduce_sum(tf.reshape(
            tf.cast(
                tf.reduce_all(preds > 0, keepdims=True),
                tf.float32
            ),
            [FLAGS.batch_size, K, *preds.shape[1:]]
        ), axis=1)


def mixup(images, labels, alpha):
    b = Beta(alpha, alpha)
    l = b.sample(images.shape[0])
    l = tf.maximum(l, 1 - l)
    concat_i_l = tf.concat((images, labels), axis=-1)
    shuffled = tf.random.shuffle(concat_i_l)
    shuffled_i, shuffled_l = shuffled[..., :images.shape[-1]], shuffled[..., images.shape[-1]:]
    mixed_images = [l[i] * images[i] + (1 - l[i]) * shuffled_i[i] for i in range(images.shape[0])]
    mixed_labels = [l[i] * labels[i] + (1 - l[i]) * shuffled_l[i] for i in range(images.shape[0])]
    return mixed_images, mixed_labels