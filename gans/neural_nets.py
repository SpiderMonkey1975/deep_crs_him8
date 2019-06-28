from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.layers import Dropout, LeakyReLU, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

def create_discriminator( num_filters ):
    """Create a simple CNN classification network.

       Parameters:
       num_filters(int): number of convolutional filters used in the first CNN layer of the network

    """
    model = Sequential()
    model.add( Conv2D(num_filters, 5, strides=2, padding='same', input_shape=(400,400,1)) )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )

    model.add( Conv2D(2*num_filters, 5, strides=2, padding='same') )
    model.add( LeakyReLU(alpha=0.2) )
    model.add( Dropout(0.3) )

    model.add( Flatten() )
    model.add( Dense(units=1,activation='sigmoid') )
    model.compile( loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'] )
    return model

def autoencoder( num_filters, num_layers, weights_file ):
    """Create a simple encoder-decoder neural network.

       Parameters:
       num_filters(int): number of convolutional filters used in the first CNN layer of the network
       num_layers(int): number of layers in the encoding (and decoding) portion of the network
       weights_file(str): name of model weights file used for a restart.  Set to none if no restart is desired.

    """
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

    model = Model( inputs=input_layer, outputs=net )
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

    if weights_file != 'none':
       model.load_weights( weights_file )
    return model

def create_gan(num_filters, num_layers, weights_file, num_gpus):
    """Create a Generative Adversarial Network.

       Parameters:
       num_filters(int): number of convolutional filters used in the first CNN layer of the generator network
       num_layers(int): number of layers in the encoding (and decoding) portion of the generator network
       weights_file(str): name of model weights file used for a restart of the generator network.  Set to none if no restart is desired.
       num_gpus(int): number of GPUs to be used in the training of the GAN

    """
    discriminator = create_discriminator( num_filters )
    generator = autoencoder( num_filters, num_layers, weights_file )

    discriminator.trainable = False
    gan_input = Input(shape = (400,400,3,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    if num_gpus>1:
       with tf.device("/cpu:0"):
            gan = Model(inputs=gan_input, outputs=gan_output)
            gan = multi_gpu_model( gan, gpus=num_gpus )
    else:
       gan = Model(inputs=gan_input, outputs=gan_output)

    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan, generator, discriminator

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
