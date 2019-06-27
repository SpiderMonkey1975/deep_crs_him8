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

def autoencoder( num_filters, num_gpus, num_layers ):
    input_layer = Input(shape = (400, 400, 3))
    net = BatchNormalization(axis=3)( input_layer )

    if num_layers>5:
       num_layers = 5

    # Encoding section
    for n in range(num_layers):
        filter_cnt = num_filters * (2**n)
        if n == 0 or n == (num_layers-1):
           kernel_size = 5
        else:
           kernel_size = 3
        net = Conv2D( filter_cnt, kernel_size, strides=2, activation='relu', padding='same')( net )
        net = BatchNormalization(axis=3)( net )

    # Decoding section
    for n in range(num_layers):
        filter_cnt = num_filters * (2**(num_layers-1-n))
        if n == 0:
           kernel_size = 5
        elif n == (num_layers-1):
           kernel_size = 5
           filter_cnt = 1
        else:
           kernel_size = 3
        net = Conv2DTranspose( filter_cnt, kernel_size, strides=2, activation='relu', padding='same')( net )
        if n != (num_layers-1):
           net = BatchNormalization(axis=3)( net )

#    net = BatchNormalization(axis=3)( input_layer )
#    net = Conv2D( num_filters, 5, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2D( num_filters*2, 3, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2D( num_filters*4, 3, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2D( num_filters*8, 5, strides=2, activation='relu', padding='same')(net)

#    net = BatchNormalization(axis=3)( net )
#    net = Conv2DTranspose(num_filters*4, 5, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2DTranspose(num_filters*2, 3, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2DTranspose(num_filters, 3, strides=2, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )
#    net = Conv2DTranspose(1, 5, strides=2, activation='relu', padding='same')(net)

    if num_gpus>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=net )
            model = multi_gpu_model( model, gpus=num_gpus )
    else:
       model = Model( inputs=input_layer, outputs=net )
    return model

##
## unet
##
## Autoencoder neural network based on the U-Net design
##------------------------------------------------------

def unet( num_filters, num_gpus ):
    input_layer = Input(shape = (400, 400, 3))

    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv1)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv2)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv3)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*8, 3, strides=1, activation='relu', padding='same')(net)
    conv4 = Conv2D(num_filters*8, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv4)
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters*16, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*16, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv4],axis=3 )
    net = Conv2D(num_filters*8, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*8, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*4, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters*2, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    if num_gpus>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=net )
            model = multi_gpu_model( model, gpus=num_gpus )
    else:
       model = Model( inputs=input_layer, outputs=net )

    return model
