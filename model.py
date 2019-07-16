from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dropout, Convolution2D, UpSampling2D, Add


def resnet50_fcn(n_classes):
    # load ResNet
    input_tensor = Input(shape=(None, None, 3))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    # add classifier
    x = base_model.get_layer('activation_48').output
    x = Dropout(0.5)(x)
    x = Convolution2D(n_classes, 1, name='pred_32', padding='valid')(x)

    # add upsampler
    stride = 32
    x = UpSampling2D(size=(stride, stride))(x)
    pred_32s = Convolution2D(n_classes, 5, name='pred_32s', padding='same')(x)

    # add 16s classifier
    x = base_model.get_layer('activation_40').output
    x = Dropout(0.5)(x)
    x = Convolution2D(n_classes, 1, name='pred_16', padding='valid')(x)
    x = UpSampling2D(name='upsampling_16', size=(stride, stride))(x)
    x = Convolution2D(n_classes, 5, name='pred_up_16', padding='same')(x)

    # merge classifiers
    pred_16s = Add()([x, pred_32s])

    # add 8s classifier
    x = base_model.get_layer('activation_22').output
    x = Dropout(0.5)(x)
    x = Convolution2D(n_classes, 1, name='pred_8', padding='valid')(x)
    x = UpSampling2D(name='upsampling_8', size=(stride/2, stride/2))(x)
    x = Convolution2D(n_classes, 5, name='pred_up_8', padding='same')(x)

    # merge classifiers
    x = Add()([x, pred_16s])

    model = Model(inputs=base_model.input, outputs=x)
    return model
