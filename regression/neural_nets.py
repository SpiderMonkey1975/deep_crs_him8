import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Reshape, multiply, add, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.regularizers import l2


def construct_model( input_layer, output_layer, num_gpu ):
    if num_gpu>1:
       with tf.device("/cpu:0"):
            model = Model( inputs=input_layer, outputs=output_layer )
            model = multi_gpu_model( model, gpus=num_gpu )
    else:
       model = Model( inputs=input_layer, outputs=output_layer )
    return model

##
##-------------  Basic Encoder-Decoder stuff ----------------------------------------------
##

def autoencoder( num_filters, num_gpus, num_layers ):
    ''' Python function that defines a simple encoder-decoder neural network design

        INPUT: num_filters -> # of filters in the first convolutional layer
               num_gpus    -> # of GPUs used in training the network
               num_layers  -> # of encoding/decoding layers in the autoencoder
    '''
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

    return construct_model( input_layer, net, num_gpus ) 

##
##-------------  U-Net stuff ----------------------------------------------
##

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

def unet( num_filters, num_gpus ):
    ''' Python function that defines U-Net architecture of an autoencoder neural network

        INPUT: num_filters -> # of filters in the first convolutional layer
               num_gpus    -> # of GPUs used in training the network
    '''
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

    return construct_model( input_layer, net, num_gpus ) 

##
##-------------  FC-DenseNet stuff ----------------------------------------------
##


def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

def SoftmaxLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """
    l = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
    l = Activation('sigmoid')(l)#or softmax for multi-class
    return l

def Tiramisu(
        input_shape=(None,None,3),
        n_classes = 1,
        n_filters_first_conv = 48,
        n_pool = 5,
        growth_rate = 16 ,
        n_layers_per_block = [4,5,7,10,12,15,12,10,7,5,4],
        dropout_p = 0.2
        ):
    if type(n_layers_per_block) == list:
            print(" ")
    elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

#####################
# First Convolution #
#####################
    inputs = Input(shape=input_shape)
    stack = Conv2D(filters=n_filters_first_conv, kernel_size=3, padding='same', kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

#####################
# Downsampling path #
#####################
    skip_connection_list = []

    for i in range(n_pool):
        for j in range(n_layers_per_block[i]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            stack = concatenate([stack, l])
            n_filters += growth_rate

        skip_connection_list.append(stack)
        stack = TransitionDown(stack, n_filters, dropout_p)
    skip_connection_list = skip_connection_list[::-1]


#####################
#    Bottleneck     #
#####################
    block_to_upsample=[]

    for j in range(n_layers_per_block[n_pool]):
        l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
        block_to_upsample.append(l)
        stack = concatenate([stack,l])
    block_to_upsample = concatenate(block_to_upsample)


#####################
#  Upsampling path  #
#####################
    for i in range(n_pool):
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i ]
        stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep)

        block_to_upsample = []
        for j in range(n_layers_per_block[ n_pool + i + 1 ]):
            l = BN_ReLU_Conv(stack, growth_rate, dropout_p=dropout_p)
            block_to_upsample.append(l)
            stack = concatenate([stack, l])
        block_to_upsample = concatenate(block_to_upsample)

#####################
#  Softmax          #
#####################
    output = SoftmaxLayer(stack, n_classes)
   
    num_gpus = 1
    return construct_model( inputs, output, num_gpus ) 
