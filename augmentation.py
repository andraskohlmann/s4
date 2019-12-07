import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS


# @tf.function
def augment(images, labels):
    batch_size = tf.shape(images)[0]

    labels = tf.expand_dims(labels, -1)
    labels = tf.cast(labels, tf.float32)
    # Crop
    boxes = tf.transpose(
        tf.stack([
            tf.random.uniform([batch_size]) * FLAGS.crop - FLAGS.crop_offset,
            tf.random.uniform([batch_size]) * FLAGS.crop - FLAGS.crop_offset,
            1 - tf.random.uniform([batch_size]) * FLAGS.crop + FLAGS.crop_offset,
            1 - tf.random.uniform([batch_size]) * FLAGS.crop + FLAGS.crop_offset
        ]),
        tf.constant([1, 0])
    )
    box_indices = tf.range(batch_size)
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
    flip_mask = tf.reshape(tf.less(tf.random.uniform([batch_size]), FLAGS.flip_prob), [-1, 1, 1, 1])
    images = tf.where(flip_mask, tf.image.flip_left_right(images), images)
    labels = tf.where(flip_mask, tf.image.flip_left_right(labels), labels)
    labels = tf.squeeze(labels, -1)
    labels = tf.cast(labels, tf.int32)

    return images, labels


@tf.function
def augment_image(images, K=1):
    images = tf.tile(images, [K, 1, 1, 1])
    batch_size = tf.shape(images)[0]

    # Crop
    boxes = tf.transpose(
        tf.stack([
            tf.random.uniform([batch_size]) * FLAGS.crop - FLAGS.crop_offset,
            tf.random.uniform([batch_size]) * FLAGS.crop - FLAGS.crop_offset,
            1 - tf.random.uniform([batch_size]) * FLAGS.crop + FLAGS.crop_offset,
            1 - tf.random.uniform([batch_size]) * FLAGS.crop + FLAGS.crop_offset
        ]),
        [1, 0]
    )
    box_indices = tf.range(batch_size)
    crop_size = FLAGS.resolution
    images = tf.image.crop_and_resize(
        images,
        boxes,
        box_indices,
        crop_size,
        method='bilinear'
    )

    # Flip
    flip_mask = tf.reshape(tf.less(tf.random.uniform([batch_size]), FLAGS.flip_prob), [-1, 1, 1, 1])
    images = tf.where(flip_mask, tf.image.flip_left_right(images), images)

    return images, boxes, flip_mask


@tf.function
def augment_labels(labels, boxes, flip_mask):
    batch_size = tf.shape(labels)[0]

    # Crop
    box_indices = tf.range(batch_size)
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
    labels = tf.where(flip_mask, tf.image.flip_left_right(labels), labels)

    return labels


@tf.function
def reverse_augment_labels(labels, boxes, flip_mask):
    batch_size = tf.shape(labels)[0]

    # Flip
    labels = tf.where(flip_mask, tf.image.flip_left_right(labels), labels)

    # Crop
    box_indices = tf.range(batch_size)
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


@tf.function
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


@tf.function
def sharpen(p):
    p_t = tf.pow(p, 1 / FLAGS.temperature)
    zero_mask = tf.reduce_sum(p_t, axis=-1, keepdims=True) > 0
    return tf.where(
        condition=zero_mask,
        x=p_t / tf.reduce_sum(p_t, axis=-1, keepdims=True),
        y=tf.zeros_like(p_t)
    )


@tf.function
def average_preds(preds, K, all=False):
    mask = tf.reduce_sum(preds, axis=-1, keepdims=True)
    # if all:
    #     mask = tf.reduce_all(mask, axis=0, keepdims=True)
    mask_reshaped = tf.reduce_sum(tf.reshape(tf.cast(
        mask, tf.float32),
        [K, FLAGS.unlabeled_batch_size, *mask.shape[1:]]), axis=0)
    return tf.reduce_sum(
        tf.reshape(preds, [K, FLAGS.unlabeled_batch_size, *preds.shape[1:]]),
        axis=0
    ) / tf.maximum(1., mask_reshaped)


@tf.function
def mixup(sampler, images, labels):
    l = sampler.sample(images.shape[0])
    l = tf.maximum(l, 1 - l)
    concat_i_l = tf.concat((images, labels), axis=-1)
    shuffled = tf.gather(concat_i_l, tf.random.shuffle(tf.range(concat_i_l.shape[0])))
    # shuffled = tf.random.shuffle(concat_i_l)
    shuffled_i, shuffled_l = shuffled[..., :images.shape[-1]], shuffled[..., images.shape[-1]:]
    mixed_images = tf.stack([l[i] * images[i] + (1 - l[i]) * shuffled_i[i] for i in range(images.shape[0])])
    mixed_labels = tf.stack([l[i] * labels[i] + (1 - l[i]) * shuffled_l[i] for i in range(images.shape[0])])
    return mixed_images, mixed_labels, shuffled_i, shuffled_l
