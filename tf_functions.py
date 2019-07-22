import tensorflow as tf
from tensorflow.python.platform import flags
from utils import debug_plot

from tqdm import tqdm
import time

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


def consistence_loss(predictions, params):
    augmented_preds = tf.unstack(predictions[1:])
    loss = 0.
    for i, pred in enumerate(augmented_preds):
        augmented = predictions[0]
        if params['flip'][i + 1] == 1:
            augmented = tf.image.flip_left_right(augmented)
        loss += tf.reduce_mean(tf.losses.mean_absolute_error(augmented, pred))
    return loss


def valid_mask(labels, preds):
    valid_indices = tf.not_equal(labels, 255)
    valid_labels = labels[valid_indices]
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


# @tf.function
def train_loop(model, optimizer, train_dataset, avg_loss, mIoU, iters):
    i = 0
    b = 0
    for images, labels in tqdm(train_dataset, total=iters):
        with tf.device('/GPU:0'):
            images, labels = augment(images, labels)
        with tf.GradientTape() as tape:
            logits = model(images)
            preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
            valid_labels, valid_logits = valid_mask(labels, logits)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
        if 0 < FLAGS.debug_freq <= b:
            debug_plot(images, labels, preds, i, b)
            b = 0
        else:
            b += 1
        i += 1


# @tf.function
def val_loop(model, val_dataset, avg_loss, mIoU, iters):
    for images, labels in tqdm(val_dataset, total=iters):
        with tf.device('/GPU:0'):
            images, labels = resize(images, labels)
        logits = model(images)
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        valid_labels, valid_logits = valid_mask(labels, logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
        loss = tf.reduce_mean(loss)
        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
