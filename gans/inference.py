from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import argparse
from neural_nets import autoencoder

from plotting_routines import plot_images

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--num_layers', type=int, default=3, help="set number of encoding/decoding layers for the generator network")
parser.add_argument('-f', '--num_filter', type=int, default=8, help="set initial number of filters used in CNN layers for the generator networks")
parser.add_argument('-w', '--weights_file', type=str, default='generator_weights.h5', help="set name of weights file for trained generator network")
args = parser.parse_args()

num_gpu = 1

##
## Read in the test datasets 
##

satellite_test_data = np.load( "../input/input_3layer_test.npy" )
crs_model_output = np.load( "../input/crs_test.npy" )

##
## Reconstruct the generator (autoencoder) neural network. Generate predicted precipitation images 
##

generator = autoencoder( args.num_filter, args.num_layers, args.weights_file )
predicted_precip = generator.predict( satellite_test_data, verbose=0 )


print(" ")
print(" ")
print("=====================================================================================")
print("                   Precipitation GAN-Optimized Regression Network")
print("=====================================================================================")
print(" ")
print("   3 channels of satellite reflectance data used")
print("   %2d filters used in the first CNN layer of Generator" % args.num_filter)
print("   %1d encoding/decoding layers used in Generator" % args.num_layers )
print(" ")

##
## Output a visual comparison between the predicted precipitation and the CRS model output
##

plot_images( crs_model_output[ :5,:,: ], 
             predicted_precip[ :5,:,:,: ], 
             args.num_filter, 
             -1 )

##
## Perform a pixel-by-pixel accuracy check. Determine # of pixels with less than 5% difference
## between the CRS Model and autoencoder outputs
##

print("   PREDICTION ACCURACY:")

predicted_precip = np.squeeze( predicted_precip )
tmp = np.absolute( crs_model_output - predicted_precip )

cnt = (tmp < 0.05).sum()
accuracy = (cnt / 1600.0) / np.float(crs_model_output.shape[0]) 
print("   within  5 percent tolerance - %5.2f" % accuracy)

cnt = (tmp < 0.1).sum()
accuracy = (cnt / 1600.0) / np.float(crs_model_output.shape[0]) 
print("   within 10 percent tolerance - %5.2f" % accuracy)

cnt = (tmp < 0.2).sum()
accuracy = (cnt / 1600.0) / np.float(crs_model_output.shape[0]) 
print("   within 20 percent tolerance - %5.2f" % accuracy)

print(" ")

