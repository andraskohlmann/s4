import os

import tensorflow as tf
from tensorflow.python.platform import flags

from data import cityscapes
from model import resnet50_fcn
from tf_functions import train_loop, val_loop
from utils import allow_growth, checkpoints

allow_growth()

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('limit', -1, 'Limit')
flags.DEFINE_integer('epoch', 10, 'Epoch number')
flags.DEFINE_integer('debug_freq', -1, 'Debug output freq')
flags.DEFINE_list('resolution', ['128', '256'], 'Resolution')
flags.DEFINE_string('input', '/Users/metuoku/data/cityscapes/', 'Cityscapes input folder')
flags.DEFINE_string('run', 'default', 'Experiment run')
flags.DEFINE_boolean('cont', False, 'Continue training from ckpt')

FLAGS.resolution = [int(_) for _ in FLAGS.resolution]
num_classes = 19

train_dataset, train_size = cityscapes(
    FLAGS.input,
    state='train',
    resize_dims=FLAGS.resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
val_dataset, val_size = cityscapes(
    FLAGS.input,
    state='val',
    resize_dims=FLAGS.resolution,
    batch_size=FLAGS.batch_size,
    limit=FLAGS.limit
)
fcn = resnet50_fcn(n_classes=num_classes)
adam = tf.keras.optimizers.Adam()
b = 0

# tb logs
train_summary_writer = tf.summary.create_file_writer('{}/{}/train'.format(FLAGS.log_dir, FLAGS.run))
val_summary_writer = tf.summary.create_file_writer('{}/{}/val'.format(FLAGS.log_dir, FLAGS.run))
if not os.path.exists(os.path.join('out', FLAGS.run)):
    os.makedirs(os.path.join('out', FLAGS.run), exist_ok=True)

# checkpointing
ckpt, manager, init_epoch = checkpoints(adam, fcn)
# min_val_loss = np.inf
max_miou = 0

# metrics
avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
mIoU = tf.keras.metrics.MeanIoU(num_classes=num_classes, dtype=tf.float32)

for i in range(init_epoch, init_epoch + FLAGS.epoch):
    with train_summary_writer.as_default():
        print('train epoch ', i)
        train_loop(fcn, adam, train_dataset, avg_loss, mIoU, iters=train_size // FLAGS.batch_size)
        tf.summary.scalar('loss', avg_loss.result(), step=i)
        tf.summary.scalar('mIoU', mIoU.result(), step=i)
        avg_loss.reset_states()
        mIoU.reset_states()

    with val_summary_writer.as_default():
        print('val epoch ', i)
        val_loop(fcn, val_dataset, avg_loss, mIoU, iters=val_size // FLAGS.batch_size)
        ckpt.step.assign_add(1)

        tf.summary.scalar('loss', avg_loss.result(), step=i)
        tf.summary.scalar('mIoU', mIoU.result(), step=i)

        if mIoU.result() > max_miou:
            manager.save()
            max_miou = mIoU.result()

        avg_loss.reset_states()
        mIoU.reset_states()
