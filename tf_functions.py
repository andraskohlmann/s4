import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def augment(image):
    flipped = tf.image.flip_left_right(image)
    images = tf.stack([image, flipped])
    params = {
        'flip': [0, 1]
    }
    return images, params


def consistence_loss(predictions, params):
    augmented_preds = tf.unstack(predictions[1:])
    loss = 0.
    for i, pred in enumerate(augmented_preds):
        augmented = predictions[0]
        if params['flip'][i + 1] == 1:
            augmented = tf.image.flip_left_right(augmented)
        loss += tf.reduce_mean(tf.losses.mean_absolute_error(augmented, pred))
    return loss


def valid_mask(labels, preds):
    valid_indices = tf.not_equal(labels, 255)
    valid_labels = labels[valid_indices]
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


@tf.function
def train_loop(model, optimizer, train_dataset, avg_loss, mIoU):
    # i = 0
    # b = 0
    for images, labels in train_dataset:
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
        # if 0 < FLAGS.debug_freq < b:
        #     debug_plot(images, valid_lbls, valid_preds, i, b)
        #     b = 0
        # else:
        #     b += 1
        # i += 1


@tf.function
def val_loop(model, val_dataset, avg_loss, mIoU):
    for images, labels in val_dataset:
        logits = model(images)
        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        valid_labels, valid_logits = valid_mask(labels, logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
        loss = tf.reduce_mean(loss)
        avg_loss.update_state(loss)
        valid_lbls, valid_preds = valid_mask(labels, preds)
        mIoU.update_state(valid_lbls, valid_preds)
