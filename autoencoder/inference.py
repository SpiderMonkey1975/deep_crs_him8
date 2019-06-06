from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import argparse, neural_nets 

from plotting_routines import compare_images

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=32, help="set initial number of filters used in CNN layers for the neural networks")
args = parser.parse_args()

num_gpu = 1

##
## Read in the input (X,Y) datasets
##----------------------------------
x = np.load( "../input/input_3layer_test.npy" )

##
## Perform inference testing using the basic autoencoder neural network design
##-----------------------------------------------------------------------------
model = neural_nets.autoencoder( args.num_filter, num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

weights_file = 'model_weights_basic_autoencoder_' + str(args.num_filter) + 'filters.h5'
model.load_weights( weights_file )

t1 = datetime.now()
basic_output = model.predict( x, batch_size=32, verbose=0 )
basic_inference_time = (datetime.now()-t1).total_seconds()

##
## Perform inference testing using the basic autoencoder neural network design
##-----------------------------------------------------------------------------
model = neural_nets.unet( args.num_filter, num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

weights_file = 'model_weights_unet_' + str(args.num_filter) + 'filters.h5'
model.load_weights( weights_file )

t1 = datetime.now()
unet_output = model.predict( x, batch_size=20, verbose=0 )
unet_inference_time = (datetime.now()-t1).total_seconds()

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   3 channels of satellite data used")
print("   %2d filters used in the first CNN layer of the neural network" % args.num_filter)
print(" ")
print("   PREDICTION TIMINGS (in seconds):")
print("   basic autoencoder - %4.3f" % basic_inference_time)
print("   u-net autoencoder - %4.3f" % unet_inference_time)
print(" ")

##
## Output a visual comparison between the two autoencoder designs and the CRS output
##
crs_output = np.load( "../input/crs_test.npy" )
compare_images( crs_output[:5,:,:], basic_output[:5,:,:], unet_output[:5,:,:], args.num_filter )
