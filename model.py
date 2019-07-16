from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dropout, Conv2D, UpSampling2D, Add

import numpy as np


def bilinear_interpolation(w):
    frac = w[0].shape[0]
    n_classes = w[0].shape[-1]
    w_bilinear = np.zeros(w[0].shape)

    for i in range(n_classes):
        w_bilinear[:, :, i, i] = 1.0 / (frac * frac) * np.ones((frac, frac))

    return w_bilinear


def resnet50_fcn(n_classes):
    # load ResNet
    # input_tensor = Input(shape=(128, 256, 3))
    input_tensor = Input(shape=(None, None, 3))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    # add classifier
    x = base_model.get_layer('activation_48').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_32', padding='same')(x)

    # add upsampler
    stride = 32
    x = UpSampling2D(size=(stride, stride))(x)
    pred_32s = Conv2D(n_classes, 5, name='pred_32s', padding='same')(x)

    # add 16s classifier
    x = base_model.get_layer('activation_39').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_16', padding='same')(x)
    x = UpSampling2D(name='upsampling_16', size=(stride / 2, stride / 2))(x)
    x = Conv2D(n_classes, 5, name='pred_up_16', padding='same')(x)

    # merge classifiers
    pred_16s = Add()([x, pred_32s])

    # add 8s classifier
    x = base_model.get_layer('activation_21').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_8', padding='same')(x)
    x = UpSampling2D(name='upsampling_8', size=(stride / 4, stride / 4))(x)
    x = Conv2D(n_classes, 5, name='pred_up_8', padding='same')(x)

    # merge classifiers
    x = Add()([x, pred_16s])

    model = Model(inputs=base_model.input, outputs=x)
    return model
