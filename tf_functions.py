import tensorflow as tf
from tensorflow.python.platform import flags
from tqdm import tqdm

from augmentation import augment, augment_image, reverse_augment_labels, resize, sharpen, augment_labels, mixup, \
    average_preds
from utils import debug_plot, plot

FLAGS = flags.FLAGS


def valid_mask(labels, preds, mask_value=255):
    valid_indices = tf.not_equal(labels, mask_value)
    valid_labels = labels[valid_indices]
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


def nonzero_preds(labels, preds):
    valid_indices = tf.reduce_sum(preds, axis=-1) > 0
    valid_labels = labels[valid_indices]
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


# @tf.function
def supervised_train_loop(model, optimizer, train_dataset, avg_loss, mIoU, iters):
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
def semisupervised_train_loop(model, optimizer, train_dataset, avg_loss, mIoU, iters):
    l = 100
    i = 0
    b = 0
    K = 2
    T = .5
    alpha = 0.75
    for X, U in tqdm(train_dataset, total=iters):
        images, labels = X
        unlabeled_images = U
        # with tf.device('/GPU:0'):
        images, labels = augment(images, labels)
        unlabeled_images, boxes, flip_mask = augment_image(unlabeled_images, K)
        # L = augment_labels(labels, boxes, flip_mask)
        U_logits = model(unlabeled_images)
        U_logits = tf.stop_gradient(tf.nn.softmax(U_logits))
        U_logits_reversed = reverse_augment_labels(U_logits, boxes, flip_mask)
        U_average = average_preds(U_logits_reversed, K)
        U_sharp = sharpen(U_average, T)
        U_sharp = tf.tile(U_sharp, [K, 1, 1, 1])
        U_final = augment_labels(U_sharp, boxes, flip_mask)

        i_c = tf.concat([images, unlabeled_images], axis=0)
        label_num = U_final.shape[-1]
        one_hot_labels = tf.one_hot(labels, depth=label_num)
        l_c = tf.concat([one_hot_labels, U_final], axis=0)
        # i_mix, l_mix = mixup(i_c, l_c, alpha)
        i_mix, l_mix = i_c, l_c
        l_labeled = tf.stack(l_mix[:images.shape[0]])
        l_unlabeled = tf.stack(l_mix[images.shape[0]:])

        with tf.GradientTape() as tape:
            logits = model(i_mix)
            logits_labeled = logits[:images.shape[0]]
            logits_unlabeled = tf.nn.softmax(logits[images.shape[0]:])

            loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=l_labeled, logits=logits_labeled)
            loss_s = tf.reduce_mean(loss_s)
            loss_unsup = tf.square(logits_unlabeled - l_unlabeled)
            loss_unsup = tf.reduce_mean(loss_unsup)
            # loss = loss_s + l * loss_unsup
            loss = loss_unsup
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        avg_loss.update_state(loss)
        preds = tf.argmax(model(images), axis=-1)
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
        # with tf.device('/GPU:0'):
        images, labels = resize(images, labels)
        logits = model(images)
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        valid_labels, valid_logits = valid_mask(labels, logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
        loss = tf.reduce_mean(loss)
        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
