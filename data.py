import glob
import tensorflow as tf


def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_string), tf.float32)
    image = tf.image.resize(image, [192, 192])
    image = image / 255.
    return image


def images_from_folder(images_url, batch_size):
    filenames = tf.constant(glob.glob(images_url))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function).batch(batch_size).repeat()
    return dataset
