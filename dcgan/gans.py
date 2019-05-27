import argparse, neural_nets
import numpy as np
from random import sample

from tensorflow.keras.optimizers import RMSprop

from my_utils import plot_images
#from neural_nets.unet import generator_model, discriminator_model, adversarial_model
from neural_nets.autoencoder import generator_model, discriminator_model, adversarial_model

img_rows = 400
img_cols = 400

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=40, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3 and 10")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set learn rate for adversarial model optimizer")
args = parser.parse_args()

if args.channels != 3 and args.channels != 10:
   args.channels = 3

##
## Read in the input datasets
##

BoM_data = np.load( "../input/crs_train.npy" )
BoM_data = BoM_data.reshape( -1, img_rows, img_cols, 1 ).astype(np.float32)

filename = "../input/input_" + str(args.channels) + "layer_train.npy"
reflectance_data = np.load( filename )

##
## Define the generator and discriminator neural networks.  Construct the 
## corresponding discriminator and adversarial models.
##

depth = 64
GM = generator_model( img_rows, img_cols, depth )

DM = discriminator_model( img_rows, img_cols, depth*2 )
DM.compile( loss='binary_crossentropy', optimizer=RMSprop(lr=args.learn_rate*2.0,decay=6e-8), metrics=['accuracy'] )

AM = adversarial_model( img_rows, img_cols, depth )
AM.compile( loss='binary_crossentropy', optimizer=RMSprop(lr=args.learn_rate,decay=3e-8), metrics=['accuracy'] )

##
## Construct the label tensors for the GANs training
##

labels = np.zeros( [2*args.batch_size,1] )
labels[ :args.batch_size,: ] = 1

##
## Perform batch-wise training over the GANs
##

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall GANs Prediction")
print("=====================================================================================")
print(" ")
print("   %1d channels of satellite data used" % args.channels)
print("   batch size of %3d images used" % args.batch_size)
print("   adversarial learn rate of %6.5f used" % args.learn_rate)
print(" ")

#for channel_no in range( args.channels ):
for channel_no in range( 1 ):
    print("   Channel %1d processing..." % (channel_no))

    cnt = 0
    for i in range(0,BoM_data.shape[0],args.batch_size):
        if ( (i+args.batch_size)>=BoM_data.shape[0] ):
           break

        idx = np.arange( i,i+args.batch_size )
        np.random.shuffle( idx ) 

        x = np.expand_dims( reflectance_data[ idx,:,:,channel_no ], axis=3 )
        images_fake = GM.predict( x )
        features = np.concatenate((images_fake, BoM_data[idx,:,:,:]))

        d_loss = DM.train_on_batch( features, labels )
        a_loss = AM.train_on_batch( x, labels[args.batch_size:] )

        if cnt == 10:
           log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
           log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
           print(log_mesg)
           cnt = 0
        cnt = cnt + 1

##
## Save the trained generator model
##

GM.save( "generator_model_save.h5" )

##
## Perform some verification testing
##

start_index = 10
idx = range( start_index,start_index+5 )

real_images = np.load( "../input/crs_test.npy" )
real_images = real_images[ idx,:,: ]

filename = "../input/input_" + str(args.channels) + "layer_test.npy"
reflectance_data = np.load( filename )
reflectance_data = reflectance_data[ idx,:,:,: ]

avg_mean = np.load( "./running_average_mean.npy" )
avg_std = np.load( "./running_average_std.npy" )

#for channel_no in range( args.channels ):
for channel_no in range( 1 ):
    x = np.expand_dims( reflectance_data[ :,:,:,channel_no ], axis=3 )
    fake_images = GM.predict( x )
    np.squeeze(fake_images).shape

    for n in range(5):
        i = start_index + n
        fake_images[n,:,:] = fake_images[n,:,:]*avg_std[i] + avg_mean[i]

    plot_images( channel_no, real_images, fake_images )

