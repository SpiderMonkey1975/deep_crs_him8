from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose

##
## cnn_valid_padding
##
## Convolutional Neural Network design with valid padding applied
##----------------------------------------------------------------

def cnn_valid_padding( num_channels ):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, num_channels)))
    model.add(Conv2D(32, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, 3, strides=2, activation='relu'))

    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(32, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, 3, strides=2, activation='relu'))
    return model

def cnn_same_padding( num_channels ):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, num_channels)))
    model.add(Conv2D(32, 5, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, 3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, 3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, 5, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, 5, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, 5, strides=2, activation='relu', padding='same'))
    return model

def cnn( num_channels, padding_type, first_last_num_filters ):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, num_channels)))
    model.add(Conv2D(32, first_last_num_filters, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, 3, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(128, 3, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(256, first_last_num_filters, strides=2, activation='relu', padding=padding_type))

    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(128, first_last_num_filters, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(64, 3, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(32, 3, strides=2, activation='relu', padding=padding_type))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2DTranspose(1, first_last_num_filters, strides=2, activation='relu', padding=padding_type))
    return model

