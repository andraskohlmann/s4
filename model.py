from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dropout, Conv2D, UpSampling2D, Add

import numpy as np


def resnet50_fcn(n_classes):
    # load ResNet
    # input_tensor = Input(shape=(128, 256, 3))
    input_tensor = Input(shape=(None, None, 3))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    # add 32s classifier
    x = base_model.get_layer('conv5_block3_out').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_32', padding='same')(x)

    # add upsampler
    stride = 32
    pred_32s = UpSampling2D(size=(stride, stride), interpolation='bilinear')(x)

    # add 16s classifier
    x = base_model.get_layer('conv4_block6_out').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_16', padding='same')(x)
    x = UpSampling2D(name='upsampling_16', size=(stride // 2, stride // 2), interpolation='bilinear')(x)

    # merge classifiers
    pred_16s = Add(name='pred_16s')([x, pred_32s])

    # add 8s classifier
    x = base_model.get_layer('conv3_block4_out').output
    x = Dropout(0.5)(x)
    x = Conv2D(n_classes, 1, name='pred_8', padding='same')(x)
    x = UpSampling2D(name='upsampling_8', size=(stride // 4, stride // 4), interpolation='bilinear')(x)

    # merge classifiers
    pred_8s = Add(name='pred_8s')([x, pred_16s])

    model = Model(inputs=base_model.input, outputs=pred_8s)
    return model
