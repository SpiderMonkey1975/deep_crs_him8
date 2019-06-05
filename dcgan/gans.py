import sys,argparse, neural_nets
import numpy as np
from random import sample

from plotting_routines import plot_images
from neural_nets.autoencoder import create_generator, create_discriminator, create_gan

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=40, help="set batch size to the GPU")
parser.add_argument('-e', '--epoch', type=int, default=5, help="set ithe number of training epochs")
parser.add_argument('-l', '--learn_rate', type=float, default=0.0001, help="set learn rate for adversarial model optimizer")
args = parser.parse_args()

##
## Read in the input datasets
##

BoM_data = np.load( "../input/crs_train.npy" )
BoM_data = BoM_data.reshape( -1, 400, 400, 1 ).astype(np.float32)
reflectance_data = np.load( "../input/input_3layer_train.npy" )

##
## Define the generator and discriminator neural networks.  Construct the 
## corresponding discriminator and adversarial models.
##

GM = create_generator( args.learn_rate )
GM.load_weights( '../autoencoder/weights_3channels_basic_autoencoder.h5' )
#GM.summary()

num_filters = 32
DM = create_discriminator( args.learn_rate, num_filters )
#DM.summary()

AM = create_gan( args.learn_rate, DM, GM )
#AM.summary()

##
## Construct the label tensors for the GANs training
##

dis_labels = np.zeros( [2*args.batch_size,1] )
dis_labels[ :args.batch_size,: ] = 0.9 

gans_labels = np.ones( [args.batch_size,1] )

##
## Perform batch-wise training over the GANs
##

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall GANs Prediction")
print("=====================================================================================")
print(" ")
print("   3 channels of satellite data used")
print("   batch size of %3d images used" % args.batch_size)
print("   adversarial learn rate of %6.5f used" % args.learn_rate)
print(" ")

real_images = np.load( "../input/crs_test.npy" )
reflectance_test_data = np.load( "../input/input_3layer_test.npy" )

cnt = 0
for epoch in range(args.epoch):
    for i in range(0,BoM_data.shape[0],args.batch_size):
        if ( (i+args.batch_size)>=BoM_data.shape[0] ):
           break

        idx = np.arange( i,i+args.batch_size )
        np.random.shuffle( idx ) 

        images_fake = GM.predict( reflectance_data[ idx,:,:,: ] )
        features = np.concatenate((images_fake, BoM_data[idx,:,:,:]))

        DM.trainable = True
        d_loss = DM.train_on_batch( features, dis_labels )

        DM.trainable = False
        a_loss = AM.train_on_batch( reflectance_data[ idx,:,:,: ], gans_labels ) 

    log_mesg = "Epoch %1d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
    log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
    print(log_mesg)

    cnt = cnt + 1
    if cnt == 100:
        cnt = 0
        gans_output = GM.predict( reflectance_test_data )
        np.squeeze(gans_output).shape
        plot_images( real_images[ :5,:,: ], gans_output[ :5,:,: ], epoch )
        print("  ** plot of real and generated images made")

##
## Perform some verification testing
##

gans_output = GM.predict( reflectance_test_data )
np.squeeze(gans_output).shape
plot_images( real_images[ :5,:,: ], gans_output[ :5,:,: ], args.epoch )

