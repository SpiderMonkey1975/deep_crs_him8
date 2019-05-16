import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, concatenate, Input
from tensorflow.keras.utils import multi_gpu_model

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

def generator_unet( input_layer ):
    depth = 64
    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    conv1 = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv1)
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
    conv2 = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv2)
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
    conv3 = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv3)
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
    conv4 = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)

    net = MaxPooling2D(2)(conv4)
#    net = BatchNormalization(axis=3)( net )
    net = Conv2D(depth*16, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*16, 3, strides=1, activation='relu', padding='same')(net)

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv4],axis=3 )
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*8, 3, strides=1, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv3],axis=3 )
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*4, 3, strides=1, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv2],axis=3 )
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth*2, 3, strides=1, activation='relu', padding='same')(net)
#    net = BatchNormalization(axis=3)( net )

    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv1],axis=3 )
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(depth, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    return net

def adversarial_model( img_rows, img_cols ):

    depth = 128 
    dropout = 0.4
    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = generator_unet( input_layer ) 
        
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
