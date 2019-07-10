import glob
import tensorflow as tf


def _parse_function(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.convert_image_dtype(tf.image.decode_jpeg(image_string), tf.float32)
    image_decoded = image_decoded / 255.
    return image_decoded


def images_from_folder(images_folder, batch_size):
    filenames = tf.constant(glob.glob(images_folder + '/*.jpg'))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function).batch(batch_size).repeat()
    return dataset
