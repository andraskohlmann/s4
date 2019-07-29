import glob
import os

import tensorflow as tf


def parser_wrapper(resize_dims):
    def _parse_function(img_filename, label_filename):
        image_string = tf.io.read_file(img_filename)
        image = tf.dtypes.cast(tf.image.decode_png(image_string), tf.float32)
        # image = tf.image.resize(image, resize_dims)
        image = image / 255.

        label_string = tf.io.read_file(label_filename)
        label = tf.dtypes.cast(tf.image.decode_png(label_string), tf.int32)
        # label = tf.image.resize(label, resize_dims, method=ResizeMethod.NEAREST_NEIGHBOR)
        label = tf.squeeze(label, -1)

        return image, label

    return _parse_function


def cityscapes(data_url, state, resize_dims, batch_size, limit=-1):
    img_filenames = sorted(glob.glob(os.path.join(data_url, 'leftImg8bit', state, '*', '*.png')))
    labels_filenames = sorted(glob.glob(os.path.join(data_url, 'gtFine', state, '*', '*_trainIds.png')))
    img_filenames_t = tf.data.Dataset.from_tensor_slices(tf.constant(img_filenames))
    labels_filenames_t = tf.data.Dataset.from_tensor_slices(tf.constant(labels_filenames))
    img_labels_filenames = tf.data.Dataset.zip((img_filenames_t, labels_filenames_t))
    dataset_size = len(img_filenames) if limit == -1 else limit
    img_labels_filenames = img_labels_filenames.take(limit).shuffle(len(img_filenames))

    dataset = img_labels_filenames.map(
        parser_wrapper(resize_dims=resize_dims),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, dataset_size
