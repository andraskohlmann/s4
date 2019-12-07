import tensorflow as tf
from tensorflow.python.ops.distributions.beta import Beta
from tensorflow.python.platform import flags
from tqdm import tqdm

from augmentation import augment, augment_image, reverse_augment_labels, resize, sharpen, augment_labels, mixup, \
    average_preds
from utils import debug_plot, plot

FLAGS = flags.FLAGS


@tf.function
def valid_mask(labels, mask_value=255):
    valid_indices = tf.not_equal(labels, mask_value)
    valid_labels = labels[valid_indices]
    return valid_labels, valid_indices


@tf.function
def valid_mask_preds(labels, preds, mask_value=255):
    valid_labels, valid_indices = valid_mask(labels, mask_value)
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


@tf.function
def nonzero_one_hot_mask(labels):
    valid_indices = tf.reduce_sum(labels, axis=-1) > 0
    valid_labels = labels[valid_indices]
    return valid_labels, valid_indices


# @tf.function
def supervised_train_loop(model, optimizer, train_dataset, avg_loss, mIoU, iters):
    i = 0
    b = 0
    for images, labels in tqdm(train_dataset, total=iters):
        # with tf.device('/GPU:0'):
        images, labels = augment(images, labels)
        with tf.GradientTape() as tape:
            logits = model(images)
            preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
            valid_labels, valid_logits = valid_mask_preds(labels, logits)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask_preds(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
        if 0 < FLAGS.debug_freq <= b:
            debug_plot(images, labels, preds, i, b)
            b = 0
        else:
            b += 1
        i += 1


@tf.function
def semisupervised_train_loop(model, optimizer, train_dataset, avg_loss, mIoU, iters):
    i = 0
    b = 0
    beta_distribution = Beta(FLAGS.alpha, FLAGS.alpha)
    for X, U in tqdm(train_dataset, total=iters):
        images, labels = X
        unlabeled_images = U
        # with tf.device('/GPU:0'):
        images, labels = augment(images, labels)
        unlabeled_images, boxes, flip_mask = augment_image(unlabeled_images, FLAGS.K)
        U_final = predict_labels(FLAGS.K, boxes, flip_mask, model, unlabeled_images)

        i_c = tf.concat([images, unlabeled_images], axis=0)
        label_num = U_final.shape[-1]
        one_hot_labels = tf.one_hot(labels, depth=label_num)
        l_c = tf.concat([one_hot_labels, U_final], axis=0)
        if FLAGS.do_mixup:
            i_mix, l_mix, i_shuffled, l_shuffled = mixup(beta_distribution, i_c, l_c)
        else:
            i_mix, l_mix = i_c, l_c
        l_labeled = tf.stack(l_mix[:images.shape[0]])
        l_unlabeled = tf.stack(l_mix[images.shape[0]:])

        with tf.GradientTape() as tape:
            loss = combined_loss(i_mix, l_labeled, l_unlabeled, model)
            # loss = loss_s
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        avg_loss.update_state(loss)
        preds = tf.argmax(model(images), axis=-1)
        valid_lbls, valid_preds = valid_mask_preds(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
        # for k in range(FLAGS.batch_size + FLAGS.unlabeled_batch_size * FLAGS.K):
        #     plot(i_mix[k], "out/{}/i{}_{}.png".format(FLAGS.run, k, i))
        #
        # for k in range(FLAGS.batch_size + FLAGS.unlabeled_batch_size * FLAGS.K):
        #     plot(l_c[k], "out/{}/l0{}_{}.png".format(FLAGS.run, k, i))
        #     # plot(l_shuffled[k], "out/{}/l1{}_{}.png".format(FLAGS.run, k, i))

        # if 0 < FLAGS.debug_freq <= b:
        #     debug_plot(images, labels, preds, i, b)
        #     b = 0
        # else:
        #     b += 1
        i += 1


@tf.function
def predict_labels(K, boxes, flip_mask, model, unlabeled_images):
    U_logits = model(unlabeled_images)
    U_logits = tf.stop_gradient(tf.nn.softmax(U_logits))
    U_logits_reversed = reverse_augment_labels(U_logits, boxes, flip_mask)
    U_average = average_preds(U_logits_reversed, K)
    U_sharp = sharpen(U_average)
    U_sharp = tf.tile(U_sharp, [K, 1, 1, 1])
    U_final = augment_labels(U_sharp, boxes, flip_mask)
    return U_final


# @tf.function
def combined_loss(i_mix, l_labeled, l_unlabeled, model):
    logits = model(i_mix)
    logits_labeled = logits[:l_labeled.shape[0]]
    logits_unlabeled = tf.nn.softmax(logits[l_labeled.shape[0]:])
    # supervised loss
    valid_labels, valid_indices = nonzero_one_hot_mask(l_labeled)
    valid_logits = logits_labeled[valid_indices]
    loss_s = tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
    loss_s = tf.reduce_mean(loss_s)
    # self-supervised loss
    valid_labels_unlabeled, valid_indices_unlabeled = nonzero_one_hot_mask(l_unlabeled)
    valid_logits_unlabeled = logits_unlabeled[valid_indices_unlabeled]
    loss_selfsup = tf.square(valid_logits_unlabeled - valid_labels_unlabeled)
    loss_selfsup = tf.reduce_mean(loss_selfsup)
    loss = loss_s + FLAGS.loss_modifier * loss_selfsup
    return loss


# @tf.function
def val_loop(model, val_dataset, avg_loss, mIoU, iters):
    for images, labels in tqdm(val_dataset, total=iters):
        # with tf.device('/GPU:0'):
        images, labels = resize(images, labels)
        logits = model(images)
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        valid_labels, valid_logits = valid_mask_preds(labels, logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
        loss = tf.reduce_mean(loss)
        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask_preds(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
