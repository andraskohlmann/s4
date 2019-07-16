import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags

from data import cityscapes
from model import resnet50_fcn

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_list('resolution', ['128', '256'], 'Resolution')
flags.DEFINE_string('input', '/Users/metuoku/data/cityscapes/', 'Cityscapes input folder')

# @tf.function
def augment(image):
    flipped = tf.image.flip_left_right(image)
    images = tf.stack([image, flipped])
    params = {
        'flip': [0, 1]
    }
    return images, params


# @tf.function
def consistence_loss(predictions, params):
    augmented_preds = tf.unstack(predictions[1:])
    loss = 0.
    for i, pred in enumerate(augmented_preds):
        augmented = predictions[0]
        if params['flip'][i + 1] == 1:
            augmented = tf.image.flip_left_right(augmented)
        loss += tf.reduce_mean(tf.losses.mean_absolute_error(augmented, pred))
    return loss


# @tf.function
def train(model, batch_data, optimizer, batch_size=1):
    with tf.GradientTape() as tape:
        images, labels = batch_data
        logits = model(images)

        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        return loss, images, labels, preds


resolution = [int(_) for _ in FLAGS.resolution]
dataset = cityscapes(FLAGS.input, state='train', resize_dims=resolution, batch_size=FLAGS.batch_size, limit=1)
fcn = resnet50_fcn(n_classes=34)
adam = tf.keras.optimizers.Adam()
for i in range(10):
    b = 0
    for batch_image in dataset:
        loss, images, labels, preds = train(fcn, batch_image, adam, FLAGS.batch_size)
        print(loss)
        plt.imsave("out/{}_{}.png".format(i, b), images[0].numpy())

        plt.imsave("out/{}_{}_lab.png".format(i, b), labels[0].numpy())
        plt.imsave("out/{}_{}_pred.png".format(i, b), preds[0].numpy())
        b += 1
