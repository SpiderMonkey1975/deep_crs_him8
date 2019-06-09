from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import sys, argparse, neural_nets 
from tiramisu_net import Tiramisu

from plotting_routines import compare_images

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=32, help="set initial number of filters used in CNN layers for the neural networks")
args = parser.parse_args()

num_gpu = 1

##
## Read in the input (X,Y) datasets
##

x = np.load( "../input/input_3layer_test.npy" )

##
## Perform inference testing using the basic autoencoder neural network design
##

model = neural_nets.autoencoder( args.num_filter, num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

weights_file = 'model_weights_basic_autoencoder_' + str(args.num_filter) + 'filters.h5'
model.load_weights( weights_file )

t1 = datetime.now()
basic_output = model.predict( x, batch_size=32, verbose=0 )
basic_inference_time = (datetime.now()-t1).total_seconds()

##
## Perform inference testing using the U-Net autoencoder neural network design
##

model = neural_nets.unet( args.num_filter, num_gpu )
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

weights_file = 'model_weights_unet_' + str(args.num_filter) + 'filters.h5'
model.load_weights( weights_file )

t1 = datetime.now()
unet_output = model.predict( x, batch_size=20, verbose=0 )
unet_inference_time = (datetime.now()-t1).total_seconds()

#
## Perform inference testing using the FC-DenseNet autoencoder neural network design
##

model = Tiramisu( input_shape=(400,400,3),
                  n_filters_first_conv=args.num_filter,
                  n_pool = 2,
                  n_layers_per_block = [4,5,7,5,4] )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

weights_file = 'model_weights_tiramisu_' + str(args.num_filter) + 'filters.h5'
model.load_weights( weights_file )

t1 = datetime.now()
tiramisu_output = model.predict( x, batch_size=20, verbose=0 )
tiramisu_inference_time = (datetime.now()-t1).total_seconds()

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
print("   basic autoencoder    - %4.3f" % basic_inference_time)
print("   u-net autoencoder    - %4.3f" % unet_inference_time)
print("   tiramisu autoencoder - %4.3f" % tiramisu_inference_time)
print(" ")

##
## Output a visual comparison between the two autoencoder designs and the CRS output
##

crs_output = np.load( "../input/crs_test.npy" )

plot_filename = 'rainfall_regression_comparison_' + str(args.num_filter) + 'filters.png'
compare_images( crs_output[:5,:,:], basic_output[:5,:,:], unet_output[:5,:,:], tiramisu_output[:5,:,:], plot_filename )

plot_filename = 'rainfall_regression_model_difference__' + str(args.num_filter) + 'filters.png'
compare_images( crs_output[:5,:,:], 
                crs_output[:5,:,:] - basic_output[:5,:,:], 
                crs_output[:5,:,:] - unet_output[:5,:,:], 
                crs_output[:5,:,:] - tiramisu_output[:5,:,:], 
                plot_filename )

sys.exit()

##
## Perform a pixel-by-pixel accuracy check. Determine # of pixels with less than 5% difference
## between the CRS Model and autoencoder outputs
##

basic_output = np.squeeze(basic_output )
tmp = np.absolute( crs_output - basic_output )
cnt = (tmp < 0.05).sum()
basic_accuracy = (cnt / 1600.0) / np.float(crs_output.shape[0]) 

unet_output = np.squeeze(unet_output )
tmp = np.absolute( crs_output - unet_output )
cnt = (tmp < 0.05).sum()
unet_accuracy = (cnt / 1600.0) / np.float(crs_output.shape[0]) 

tiramisu_output = np.squeeze(tiramisu_output )
tmp = np.absolute( crs_output - tiramisu_output )
cnt = (tmp < 0.05).sum()
tiramisu_accuracy = (cnt / 1600.0) / np.float(crs_output.shape[0]) 

print("   PREDICTION ACCURACY (within 5% tolerance):")
print("   basic autoencoder    - %5.2f" % basic_accuracy)
print("   u-net autoencoder    - %5.2f" % unet_accuracy)
print("   tiramisu autoencoder - %5.2f" % tiramisu_accuracy)
print(" ")

