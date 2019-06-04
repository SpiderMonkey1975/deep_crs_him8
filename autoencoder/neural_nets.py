import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model

##
## autoencoder 
##
## Simple encoder neural network design
##----------------------------------------------------------------

def autoencoder():
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

##
## unet
##
## Autoencoder neural network based on the U-Net design
##------------------------------------------------------

def unet():
    depth = 32
    input_layer = Input(shape = (400, 400, 3))

    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv1)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv2)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv3)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
    conv4 = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv4)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*16, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*16, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv4],axis=3 )
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

  #  with tf.device("/cpu:0"):
  #       model = Model( inputs=input_layer, outputs=net )
  #       model = multi_gpu_model( model, gpus=4 )
    model = Model( inputs=input_layer, outputs=net )

    return model
