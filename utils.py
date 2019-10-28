import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
import numpy as np

from labels import label_to_image

FLAGS = flags.FLAGS


def allow_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def checkpoints(optimizer, network, max_to_keep=3):
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=network)
    manager = tf.train.CheckpointManager(ckpt, '{}/{}/tf_ckpts'.format(FLAGS.log_dir, FLAGS.run), max_to_keep=max_to_keep)
    ckpt.restore(manager.latest_checkpoint)
    init_epoch = 0
    if manager.latest_checkpoint and FLAGS.cont:
        print("Restored from {}".format(manager.latest_checkpoint))
        init_epoch = int(manager.latest_checkpoint.split('-')[-1])
    else:
        print("Initializing from scratch.")
    return ckpt, manager, init_epoch


def debug_plot(images, labels, preds, i, b):
    plt.imsave("out/{}/{}_{}.png".format(FLAGS.run, i, b), images[0].numpy())
    plt.imsave("out/{}/{}_{}_lab.png".format(FLAGS.run, i, b), labels[0].numpy())
    plt.imsave("out/{}/{}_{}_pred.png".format(FLAGS.run, i, b), preds[0].numpy())
    plt.imsave("out/{}/{}_{}2.png".format(FLAGS.run, i, b), images[1].numpy())
    plt.imsave("out/{}/{}_{}2_lab.png".format(FLAGS.run, i, b), labels[1].numpy())
    plt.imsave("out/{}/{}_{}2_pred.png".format(FLAGS.run, i, b), preds[1].numpy())


def plot(image, fname="out.png"):
    img = image.numpy()
    if len(img.shape) == 3 and img.shape[-1] != 3:
        img = img.argmax(-1)
    if len(img.shape) < 3:
        img = label_to_image(img)
    plt.imsave(fname, img)
