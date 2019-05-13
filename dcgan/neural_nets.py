from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization

def generator( img_rows, img_cols ):
    G = Sequential()

    input_shape = (img_rows, img_cols, 1)
    G.add(Conv2D(32, 5, strides=2, input_shape=input_shape, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(64, 3, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(128, 3, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(256, 5, strides=1, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2DTranspose(1, 5, padding='same'))
    G.add(Activation('sigmoid'))
    return G

def discriminator( img_rows, img_cols ):
    D = Sequential()
    depth = 32 
    dropout = 0.4
        
    input_shape = (img_rows, img_cols, 1)
    D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth*2, 3, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth*4, 3, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    return D
    