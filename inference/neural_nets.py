from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.layers import Dropout, LeakyReLU, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

def construct_model( input_layer, output_layer, num_gpu ):
    if num_gpu>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=output_layer )
            model = multi_gpu_model( model, gpus=num_gpu )
    else:
       model = Model( inputs=input_layer, outputs=output_layer )
    return model

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

##
## unet
##
## Autoencoder neural network based on the U-Net design
##------------------------------------------------------

def unet_encoder_block( net, num_filters ):
    net = MaxPooling2D(2)( net )
    net = BatchNormalization(axis=3)( net )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    return net

def unet_decoder_block( net, conv, num_filters ):
    net = UpSampling2D(2)(net)
    net = concatenate( [net,conv],axis=3 )
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    net = BatchNormalization(axis=3)( net )
    return net

def unet( num_filters, num_gpus, weights_file ):
    input_layer = Input(shape = (400, 400, 3))

    conv_list = []

    net = Conv2D( 1, 3, strides=1, activation='relu', padding='same')(input_layer)
    net = Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net)
    conv_list.append( Conv2D(num_filters, 3, strides=1, activation='relu', padding='same')(net) )

    for n in range(1,5):
        conv_list.append( unet_encoder_block( conv_list[n-1], num_filters*(2**n) ))

    net = conv_list[4]
    for n in range(3,-1,-1):
        net = unet_decoder_block( net, conv_list[n], num_filters*(2**n) )

    net = Conv2D(1, 3, strides=1, activation='relu', padding='same')(net)

    model = construct_model( input_layer, net, num_gpus )
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
    #generator = autoencoder( num_filters, num_layers, weights_file )
    generator = unet( num_filters, num_gpus, weights_file )

    discriminator.trainable = False
    gan_input = Input(shape = (400,400,3,))
    x = generator(gan_input)
    gan_output = discriminator(x)

    gan = construct_model( gan_input, gan_output, num_gpus )

    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan, generator, discriminator

