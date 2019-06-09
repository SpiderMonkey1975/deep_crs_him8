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
## Perform inference testing using the basic autoencoder neural network design
##

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

##
## Perform a pixel-by-pixel accuracy check. Determine # of pixels with less than 5% difference
## between the CRS Model and autoencoder outputs
##

basic_accuracy = 0.0
unet_accuracy = 0.0

for n in range(crs_output.shape[0]):

    cnt = 0.0
    cnt2 = 0.0

    for i in range(400):
        for j in range(400):
            if abs(crs_output[n,i,j]-basic_output[n,i,j]) < 0.05:
               cnt = cnt + 1.0
            if abs(crs_output[n,i,j]-unet_output[n,i,j]) < 0.05:
               cnt2 = cnt2 + 1.0

    basic_accuracy = basic_accuracy + (cnt/160000)
    unet_accuracy = unet_accuracy + (cnt2/160000)

basic_accuracy = (basic_accuracy / crs_output.shape[0]) * 100.0
unet_accuracy = (unet_accuracy / crs_output.shape[0]) * 100.0

print("   PREDICTION ACCURACY (within 5% tolerance):")
print("   basic autoencoder - %5.2f" % basic_accuracy)
print("   u-net autoencoder - %5.2f" % unet_accuracy)
print(" ")

basic_accuracy = 0.0
unet_accuracy = 0.0

for n in range(crs_output.shape[0]):

    cnt = 0.0
    cnt2 = 0.0

    for i in range(400):
        for j in range(400):
            if abs(crs_output[n,i,j]-basic_output[n,i,j]) < 0.05:
               cnt = cnt + 1.0
            if abs(crs_output[n,i,j]-unet_output[n,i,j]) < 0.05:
               cnt2 = cnt2 + 1.0

    basic_accuracy = basic_accuracy + (cnt/160000)
    unet_accuracy = unet_accuracy + (cnt2/160000)

basic_accuracy = (basic_accuracy / crs_output.shape[0]) * 100.0
unet_accuracy = (unet_accuracy / crs_output.shape[0]) * 100.0

print("   PREDICTION ACCURACY (within 2% tolerance):")
print("   basic autoencoder - %5.2f" % basic_accuracy)
print("   u-net autoencoder - %5.2f" % unet_accuracy)
print(" ")

