from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dropout, Convolution2D, UpSampling2D


def resnet50_fcn(n_classes):
    # load ResNet
    input_tensor = Input(shape=(None, None, 3))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    # add classifier
    x = base_model.get_layer('activation_48').output
    x = Dropout(0.5)(x)
    x = Convolution2D(n_classes, 1, 1, name='pred_32', padding='valid')(x)

    # add upsampler
    stride = 32
    x = UpSampling2D(size=(stride, stride))(x)
    x = Convolution2D(n_classes, 5, 5, name='pred_32s', padding='same')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model
