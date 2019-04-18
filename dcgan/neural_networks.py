from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, Flatten, UpSampling2D, Reshape

##
## discriminator 
##
## Neural network that determines if an input image is a real CRS
## image or nor
##----------------------------------------------------------------

def discriminator():
    model = Sequential()

    model.add( Conv2D( 32, 5, strides=2, activation='relu', input_shape=(400,400,1), padding='same' ) )
    model.add( Dropout(rate=0.4) )

    model.add( Conv2D( 64, 3, strides=2, activation='relu', padding='same' ) )
    model.add( Dropout(rate=0.4) )

    model.add( Conv2D( 128, 3, strides=2, activation='relu', padding='same' ) )
    model.add( Dropout(rate=0.4) )

    model.add( Conv2D( 256, 3, strides=2, activation='relu', padding='same' ) )
    model.add( Dropout(rate=0.4) )

    model.add( Flatten() )
    model.add( Dense(1, activation='sigmoid') )
    return model

##
## generator
##
## Neural network that inputs Himawari8 input data and generates a CRS-like image
##--------------------------------------------------------------------------------

def generator(num_channels):
    model = Sequential()

#    model.add( Flatten(input_shape=(400,400,num_channels)) )
#    model.add( Dense(400*400*num_channels,activation='relu') )
#    model.add( BatchNormalization(momentum=0.9) )
    model.add( Reshape((50,50,64*num_channels), input_shape=(400,400,num_channels)) )
    model.add( Dropout(rate=0.4) )

    model.add( UpSampling2D() )
    model.add( Conv2DTranspose(256, 5, activation='relu', padding='same') )
    model.add( BatchNormalization(axis=3,momentum=0.9) )

    model.add( UpSampling2D() )
    model.add( Conv2DTranspose(128, 5, activation='relu', padding='same') )
    model.add( BatchNormalization(axis=3,momentum=0.9) )

    model.add( UpSampling2D() )
    model.add( Conv2DTranspose(64, 5, activation='relu', padding='same') )
    model.add( BatchNormalization(axis=3,momentum=0.9) )

    model.add( Conv2DTranspose(32, 5, activation='relu', padding='same') )
    model.add( BatchNormalization(axis=3,momentum=0.9) )

    model.add( Conv2DTranspose(1, 5, activation='sigmoid', padding='same') )
    return model

