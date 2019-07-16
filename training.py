import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags

from data import cityscapes
from model import resnet50_fcn
from utils import allow_growth

allow_growth()

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('limit', -1, 'Limit')
flags.DEFINE_integer('epoch', 10, 'Epoch number')
flags.DEFINE_integer('debug_freq', -1, 'EDebug output freq')
flags.DEFINE_list('resolution', ['128', '256'], 'Resolution')
flags.DEFINE_string('input', '/Users/metuoku/data/cityscapes/', 'Cityscapes input folder')
flags.DEFINE_string('tb_dir', 'logs', 'Tensorboard')


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
train_dataset = cityscapes(
    FLAGS.input,
    state='train',
    resize_dims=resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
val_dataset = cityscapes(
    FLAGS.input,
    state='val',
    resize_dims=resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
fcn = resnet50_fcn(n_classes=34)
adam = tf.keras.optimizers.Adam()
b = 0


train_summary_writer = tf.summary.create_file_writer('{}/train'.format(FLAGS.tb_dir))
val_summary_writer = tf.summary.create_file_writer('{}/val'.format(FLAGS.tb_dir))

for i in range(FLAGS.epoch):
    with train_summary_writer.as_default():
        print('train epoch ', i)
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        for batch_image in train_dataset:
            loss, ims, labels, preds = train(fcn, batch_image, adam, FLAGS.batch_size)
            avg_loss.update_state(loss)
            if 0 < FLAGS.debug_freq < b:
                plt.imsave("out/{}_{}.png".format(i, b), ims[0].numpy())

                plt.imsave("out/{}_{}_lab.png".format(i, b), lbls[0].numpy())
                plt.imsave("out/{}_{}_pred.png".format(i, b), preds[0].numpy())
                b = 0
            else:
                b += 1
        tf.summary.scalar('loss', avg_loss.result(), step=i)
        avg_loss.reset_states()

    with val_summary_writer.as_default():
        print('val epoch ', i)
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        for batch_image in val_dataset:
            loss, images, lbls, preds = train(fcn, batch_image, adam, FLAGS.batch_size)
            avg_loss.update_state(loss)
            if 0 < FLAGS.debug_freq < b:
                plt.imsave("out/{}_{}.png".format(i, b), ims[0].numpy())

                plt.imsave("out/{}_{}_lab.png".format(i, b), lbls[0].numpy())
                plt.imsave("out/{}_{}_pred.png".format(i, b), preds[0].numpy())
                b = 0
            else:
                b += 1
        tf.summary.scalar('loss', avg_loss.result(), step=i)
        avg_loss.reset_states()
