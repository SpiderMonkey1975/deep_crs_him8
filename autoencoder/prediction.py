from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import neural_nets 

from plotting_routines import compare_images

##
## Read in the input (X,Y) datasets
##----------------------------------
x = np.load( "../input/input_3layer_test.npy" )

##
## Perform inference testing using the basic autoencoder neural network design
##-----------------------------------------------------------------------------
model = neural_nets.autoencoder()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
model.load_weights( "weights_3channels_basic_autoencoder.h5" )

t1 = datetime.now()
basic_output = model.predict( x, batch_size=32, verbose=0 )
basic_inference_time = (datetime.now()-t1).total_seconds()

##
## Perform inference testing using the basic autoencoder neural network design
##-----------------------------------------------------------------------------
model = neural_nets.unet()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
model.load_weights( "weights_3channels_unet_autoencoder.h5" )

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
print(" ")
print("   PREDICTION TIMINGS (in seconds):")
print("   basic autoencoder - %4.3f" % basic_inference_time)
print("   u-net autoencoder - %4.3f" % unet_inference_time)
print(" ")

##
## Output a visual comparison between the two autoencoder designs and the CRS output
##
crs_output = np.load( "../input/crs_test.npy" )
compare_images( crs_output[:5,:,:], basic_output[:5,:,:], unet_output[:5,:,:] )
