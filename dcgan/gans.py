import argparse, neural_nets
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input

from my_utils import plot_images


img_rows = 400
img_cols = 400

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3 and 10")
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

input_layer = Input(shape = (img_rows, img_cols, 1))
net = neural_nets.generator_unet( input_layer )
with tf.device("/cpu:0"):
     GN = Model( inputs=input_layer, outputs=net )
     GN = multi_gpu_model( GN, gpus=4 )

DM = neural_nets.discriminator_model( img_rows, img_cols )
DM.compile( loss='binary_crossentropy', optimizer=RMSprop(lr=0.0002,decay=6e-8), metrics=['accuracy'] )

AM = neural_nets.adversarial_model( img_rows, img_cols )
AM.compile( loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001,decay=3e-8), metrics=['accuracy'] )

##
## Construct the label tensors for the GANs training
##

labels = np.ones( [2*args.batch_size,1] )
labels[ :args.batch_size,: ] = 1

##
## Perform batch-wise training over the GANs
##

for channel_no in range( args.channels ):
    print("channel %1d processing..." % (channel_no))

    cnt = 0
    for i in range(0,BoM_data.shape[0],args.batch_size):
        if i+args.batch_size > BoM_data.shape[0]:
           break 

        ind = np.arange( i,i+args.batch_size )
        np.random.shuffle( ind )

        x = np.expand_dims( reflectance_data[ ind,:,:,channel_no ], axis=3 )
        images_fake = GN.predict( x )
        features = np.concatenate((BoM_data[ind,:,:,:], images_fake))

        # randomly sort the images in the input batch
        ind = np.arange( 2*args.batch_size )
        np.random.shuffle( ind )

        d_loss = DM.train_on_batch( features[ ind,:,:,: ], labels[ ind,: ] )
        a_loss = AM.train_on_batch( x, labels[args.batch_size:] )

        if cnt == 10:
           log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
           log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
           print(log_mesg)
           cnt = 0
        cnt = cnt + 1

##
## Perform some verification testing
##

BoM_data = np.load( "../input/crs_test.npy" )
BoM_data = BoM_data.reshape( -1, img_rows, img_cols, 1 ).astype(np.float32)

filename = "../input/input_" + str(args.channels) + "layer_test.npy"
reflectance_data = np.load( filename )

for channel_no in range( args.channels ):
    ind = np.random.randint(0, BoM_data.shape[0], 5)
    x = np.expand_dims( reflectance_data[ ind,:,:,channel_no ], axis=3 )

    fake_images = GN.predict( x )
    real_images = BoM_data[ ind,:,:,: ]

    for n in range(len(ind)):
        mean_val = np.mean( real_images[n,:,:,:] )
        std_dev = np.std( real_images[n,:,:,:] )
        fake_images[n,:,:,:] = fake_images[n,:,:,:]*std_dev + mean_val

    plot_images( channel_no, real_images, fake_images )

