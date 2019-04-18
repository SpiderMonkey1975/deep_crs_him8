from tensorflow.keras.optimizers import RMSprop 
import numpy as np
import argparse, neural_networks, sys, random

##----------------
## Initial Setup
##----------------

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3, 4, 5, 10")
args = parser.parse_args()

if args.channels != 3 and args.channels != 4 and args.channels != 5 and args.channels != 10:
   args.channels = 3

# read in feature and label datasets 
input_dir = "../cnn/input/"

y_file = input_dir + "crs.npy"
x_file = input_dir + "input_" + str(args.channels) + "layer.npy"

x = np.load( x_file )[:2473,:,:,:]
y_true = np.load( y_file )[:2473,:,:,None]


##--------------------------------------------
## Generate the initial "fake" label dataset
##--------------------------------------------

# create the generator model
generator = neural_networks.generator( args.channels )
generator.compile( loss='binary_crossentropy', 
                   optimizer=RMSprop(lr=0.0002,decay=3e-8), 
                   metrics=['accuracy'] )

# train it for 2 epochs
hist = generator.fit( x, y_true, batch_size=args.batch_size, verbose=0, epochs=2, validation_split=.25, shuffle=True )

# use partially trained generator to construct fake input data
y_fake = generator.predict( x, batch_size=args.batch_size, verbose=0 )

# one-hot encoding. 1 denotes a true image, 0 denotes a fake image
true_ids = random.sample( range(y_true.shape[0]), 1237 )

one_hot_encoding = np.zeros((y_true.shape[0],1), dtype=int)

feature_data = y_fake
for n in true_ids:
    one_hot_encoding[n] = 1
    feature_data[n,:,:,:] = y_true[n,:,:,:]
 
##--------------------------
## Train the discriminator 
##--------------------------

discriminator = neural_networks.discriminator()
discriminator.compile( loss='binary_crossentropy', 
                       optimizer=RMSprop(lr=0.0001,decay=6e-8), 
                       metrics=['accuracy'] )

hist = discriminator.fit( feature_data, one_hot_encoding, 
                          batch_size=args.batch_size, 
                          epochs=10, 
                          validation_split=.25, 
                          shuffle=True )

##----------------------
## Train the generator 
##----------------------

discriminator_guesses = discriminator.predict( feature_data, batch_size=args.batch_size, verbose=0 )

hist = generator.fit( x, discriminator_guesses, 
                      batch_size=args.batch_size,
                      epochs=10,
                      validation_split=.25,
                      shuffle=True )

