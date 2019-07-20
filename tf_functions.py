import tensorflow as tf


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
def valid_mask(labels, preds):
    valid_indices = tf.not_equal(labels, 255)
    valid_labels = labels[valid_indices]
    valid_preds = preds[valid_indices]
    return valid_labels, valid_preds


# @tf.function
def train(model, batch_data, optimizer):
    with tf.GradientTape() as tape:
        images, labels = batch_data
        logits = model(images)

        preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
        valid_labels, valid_logits = valid_mask(labels, logits)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
        loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        return loss, images, labels, preds


# @tf.function
def validate(model, batch_data):
    images, labels = batch_data
    logits = model(images)

    preds = tf.argmax(tf.nn.softmax(logits), axis=-1)
    valid_labels, valid_logits = valid_mask(labels, logits)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_logits)
    loss = tf.reduce_mean(loss)
    return loss, images, labels, preds
