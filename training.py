import tensorflow as tf

from data import images_from_folder
from model import resnet50_fcn


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
    loss = 0
    for i, pred in enumerate(augmented_preds):
        augmented = predictions[0]
        if params['flip'][i + 1] == 1:
            augmented = tf.image.flip_left_right(augmented)
        loss += tf.reduce_mean(tf.losses.mean_absolute_error(augmented, pred))
    return loss


# @tf.function
def train(model, dataset, optimizer, batch_size=1):
    for batch_image in dataset:
        with tf.GradientTape() as tape:
            loss = 0
            for image in batch_image:
                variations, params = augment(image)
                predictions = model(variations)
                loss += consistence_loss(predictions, params)
                # loss += confidence_loss(predictions)
            gradients = tape.gradient(loss/batch_size, model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        print(loss)


input_url = '/Users/metuoku/data/cityscapes/leftImg8bit/train/*/*.png'
batch_size = 2
dataset = images_from_folder(input_url, batch_size=batch_size)
fcn = resnet50_fcn(n_classes=8)
for i in range(10):
    train(fcn, dataset, tf.keras.optimizers.Adam(), batch_size)
