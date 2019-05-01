from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, Conv2D, Conv2DTranspose, Flatten

##
## discriminator 
##
## Neural network that determines if an input image is a real CRS
## image or nor
##----------------------------------------------------------------

def discriminator():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 1)))
    #model.add(Conv2D(32, 5, strides=2, activation='relu', padding='same'))
    model.add(Conv2D(32, 5, strides=2, padding='same'))
    model.add( LeakyReLU(alpha=0.2) )
    model.add(BatchNormalization(axis=3))
    #model.add(Conv2D(64, 3, strides=2, activation='relu', padding='same'))
    model.add(Conv2D(64, 3, strides=2, padding='same'))
    model.add( LeakyReLU(alpha=0.2) )
    model.add(BatchNormalization(axis=3))
    #model.add(Conv2D(128, 3, strides=2, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, strides=2, padding='same'))
    model.add( LeakyReLU(alpha=0.2) )
    model.add(BatchNormalization(axis=3))
    #model.add(Conv2D(256, 5, strides=2, activation='relu)', padding='same'))
    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add( LeakyReLU(alpha=0.2) )

    model.add( Flatten() )
    model.add( Dense(1, activation='sigmoid') )
    return model

##
## generator
##
## Neural network that inputs Himawari8 input data and generates a CRS-like image
##--------------------------------------------------------------------------------

def generator():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 3)))
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
