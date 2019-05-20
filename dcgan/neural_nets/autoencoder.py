import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, concatenate, Input
from tensorflow.keras.utils import multi_gpu_model

def discriminator( input_layer, depth ):
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
    return net 

def discriminator_model( img_rows, img_cols, depth ):
    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = discriminator( input_layer, depth )
    return Model( inputs=input_layer, outputs=net )


def generator( input_layer, depth ):
    net = BatchNormalization(axis=3)(input_layer)
    net = Conv2D(depth, 5, strides=2, activation='relu', padding='same')(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2D(depth*2, 3, strides=2, activation='relu', padding='same')(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2D(depth*4, 3, strides=2, activation='relu', padding='same')(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2D(depth*8, 5, strides=2, activation='relu', padding='same'))

    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose(depth*4, 5, strides=2, activation='relu', padding='same')(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose(depth*2, 3, strides=2, activation='relu', padding='same')(net)

    net = BatchNormalization(axis=3)(net)
    net = Conv2DTranspose(depth, 3, strides=2, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)(net)

    net = Conv2DTranspose(1, 5, strides=2, activation='relu', padding='same')(net)
    return net

def generator_model( img_rows, img_cols, depth ):
    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = generator( input_layer, depth )
    return Model( inputs=input_layer, outputs=net )

def adversarial_model( img_rows, img_cols, depth ):

    input_layer = Input(shape = (img_rows, img_cols, 1))
    net = generator( input_layer, depth ) 
    net = discriminator( net, depth ) 
        
    with tf.device("/cpu:0"):
         model = Model( inputs=input_layer, outputs=net )
         model = multi_gpu_model( model, gpus=4 )
    return model 
