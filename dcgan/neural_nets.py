import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, concatenate, Input
from tensorflow.keras.utils import multi_gpu_model

def generator( img_rows, img_cols ):
    G = Sequential()
    depth = 128 

    input_shape = (img_rows, img_cols, 1)
    G.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(depth, 3, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(depth, 3, strides=2, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2D(depth, 5, strides=1, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(depth, 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(depth, 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(depth, 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2DTranspose(1, 5, padding='same'))
    G.add(Activation('sigmoid'))
    return G

def discriminator( img_rows, img_cols ):
    D = Sequential()
    depth = 128 
    dropout = 0.4
        
    input_shape = (img_rows, img_cols, 1)
    D.add(Conv2D(depth, 5, strides=2, input_shape=input_shape, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth, 3, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth, 3, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth, 5, strides=1, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    return D

def discriminator_model( img_rows, img_cols ):
    depth = 128 
    dropout = 0.4
        
    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = Conv2D(depth, 5, strides=2, padding='same')(input_layer)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 3, strides=2, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 3, strides=2, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 5, strides=1, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Flatten()(net)
    net = Dense(1)(net)
    net = Activation('sigmoid')(net)
    return Model( inputs=input_layer, outputs=net ) 

def generator_unet_model( img_rows, img_cols ):
    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv1)
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv2)
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv3)
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)
    conv4 = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv4)
    net = Conv2D(1024, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1024, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv4],axis=3 )
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    with tf.device("/cpu:0"):
         model = Model( inputs=input_layer, outputs=net )
         model = multi_gpu_model( model, gpus=4 )
    return model

def adversarial_model( img_rows, img_cols ):

    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv1)
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv2)
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv3)
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)
    conv4 = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv4)
    net = Conv2D(1024, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1024, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv4],axis=3 )
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(512, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(256, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(128, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(64, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(2, 3, strides=1, activation='relu', padding='same')(net)

    depth = 128 
    dropout = 0.4
        
    net = Conv2D(depth, 5, strides=2, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 3, strides=2, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 3, strides=2, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Conv2D(depth, 5, strides=1, padding='same')(net)
    net = LeakyReLU(alpha=0.2)(net)
    net = Dropout(dropout)(net)

    net = Flatten()(net)
    net = Dense(1)(net)
    net = Activation('sigmoid')(net)
    
    with tf.device("/cpu:0"):
         model = Model( inputs=input_layer, outputs=net )
         model = multi_gpu_model( model, gpus=4 )
    return model 
