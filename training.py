import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.platform import flags
from tqdm import tqdm

from data import cityscapes
from model import resnet50_fcn
from tf_functions import train, validate
from utils import allow_growth, checkpoints

allow_growth()

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('limit', -1, 'Limit')
flags.DEFINE_integer('epoch', 10, 'Epoch number')
flags.DEFINE_integer('debug_freq', -1, 'Debug output freq')
flags.DEFINE_list('resolution', ['128', '256'], 'Resolution')
flags.DEFINE_string('input', '/Users/metuoku/data/cityscapes/', 'Cityscapes input folder')
flags.DEFINE_boolean('continue', False, 'Continue training from ckpt')

resolution = [int(_) for _ in FLAGS.resolution]
train_dataset, train_size = cityscapes(
    FLAGS.input,
    state='train',
    resize_dims=resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
val_dataset, val_size = cityscapes(
    FLAGS.input,
    state='val',
    resize_dims=resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
fcn = resnet50_fcn(n_classes=34)
adam = tf.keras.optimizers.Adam()
b = 0

# tb logs
train_summary_writer = tf.summary.create_file_writer('{}/tb/train'.format(FLAGS.log_dir))
val_summary_writer = tf.summary.create_file_writer('{}/tb/val'.format(FLAGS.log_dir))

# checkpointing
ckpt, manager = checkpoints(adam, fcn)
min_val_loss = np.inf

for i in range(FLAGS.epoch):
    with train_summary_writer.as_default():
        print('train epoch ', i)
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        for batch_image in tqdm(train_dataset, total=train_size // FLAGS.batch_size):
            loss, ims, lbls, preds = train(fcn, batch_image, adam)
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
        for batch_image in tqdm(val_dataset, total=val_size // FLAGS.batch_size):
            loss, ims, lbls, preds = validate(fcn, batch_image)
            avg_loss.update_state(loss)
            if 0 < FLAGS.debug_freq < b:
                plt.imsave("out/{}_{}.png".format(i, b), ims[0].numpy())

                plt.imsave("out/{}_{}_lab.png".format(i, b), lbls[0].numpy())
                plt.imsave("out/{}_{}_pred.png".format(i, b), preds[0].numpy())
                b = 0
            else:
                b += 1
        ckpt.step.assign_add(1)
        val_loss = avg_loss.result()
        tf.summary.scalar('loss', val_loss, step=i)

        if min_val_loss > val_loss:
            manager.save()
            min_val_loss = val_loss

        avg_loss.reset_states()
