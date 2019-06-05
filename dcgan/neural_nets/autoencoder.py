import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization

def create_generator( learn_rate ):
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

    model.compile( loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'] )
    return model

def create_discriminator( learn_rate, depth ):
    model = Sequential()
    model.add( Conv2D(depth, 5, strides=2, padding='same', input_shape=(400,400,1)) )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )
    model.add( Conv2D(depth, 3, strides=2, padding='same') )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )
    model.add( Conv2D(depth, 3, strides=2, padding='same') )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )
    model.add( Conv2D(depth, 5, strides=1, padding='same') )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )

    model.add( Flatten() )
    model.add( Dense(units=1,activation='sigmoid') )
    model.compile( loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'] )
    return model

def create_gan( learn_rate, discriminator, generator ):
    discriminator.trainable = False
    gans_input = Input(shape=(400,400,3))
    x = generator( gans_input )
    gans_output = discriminator(x)
    gans = Model(inputs=gans_input, outputs=gans_output)
    gans.compile( loss='binary_crossentropy', optimizer=Adam(lr=learn_rate), metrics=['accuracy'] )
    return gans
