import os

import tensorflow as tf
from absl import app
from absl import flags

from config import define_flags
from data import cityscapes
from model import resnet50_fcn
from tf_functions import supervised_train_loop, val_loop
from utils import allow_growth, checkpoints

FLAGS = flags.FLAGS
define_flags()


def main(argv):
    FLAGS.resolution = [int(_) for _ in FLAGS.resolution]
    num_classes = 19

    allow_growth()

    train_dataset, train_size = cityscapes(
        FLAGS.input,
        state='train',
        resize_dims=FLAGS.resolution,
        batch_size=FLAGS.batch_size,
        take=FLAGS.limit
    )
    val_dataset, val_size = cityscapes(
        FLAGS.input,
        state='val',
        resize_dims=FLAGS.resolution,
        batch_size=FLAGS.batch_size
    )
    fcn = resnet50_fcn(n_classes=num_classes)
    adam = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
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
            supervised_train_loop(fcn, adam, train_dataset, avg_loss, mIoU, iters=train_size // FLAGS.batch_size)
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


if __name__ == '__main__':
    app.run(main)
