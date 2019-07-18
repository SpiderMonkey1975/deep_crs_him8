import numpy as np
import os, sys, argparse, neural_nets 

from datetime import datetime
from plotting_routines import plot_images
from tensorflow.keras.optimizers import Adam

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--num_filter', type=int, default=8, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder, unet and tiramisu")
args = parser.parse_args()

num_gpu = 1

##
## Read in the input (X,Y) datasets
##

x = np.load( "../input/input_3layer_test.npy" )

##
## Reconstruct the appropriate autoencoder architecture 
##

if args.neural_net == 'basic_autoencoder':
   model = neural_nets.autoencoder( args.num_filter, num_gpu, 3 )
   weights_file = 'model_weights_basic_autoencoder_' + str(args.num_filter) + 'filters.h5'
elif args.neural_net == 'unet':
   model = neural_nets.unet( args.num_filter, num_gpu )
   weights_file = 'model_weights_unet_' + str(args.num_filter) + 'filters.h5'
elif args.neural_net == 'tiramisu':
   model = neural_nets.Tiramisu( input_shape=(400,400,3), n_filters_first_conv=args.num_filter, n_pool = 2, n_layers_per_block = [4,5,7,5,4] )
   weights_file = 'model_weights_tiramisu_' + str(args.num_filter) + 'filters.h5'

##
## Compile the model and load the pretrained weights
##

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

if os.path.isfile( weights_file ):
   model.load_weights( weights_file )
else:
   print("ERROR: model weights file cannot be found")
   sys.exit()

##
## Perform inference testing 
##

t1 = datetime.now()
output = model.predict( x, batch_size=32, verbose=0 )
inference_time = (datetime.now()-t1).total_seconds()

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   3 channels of satellite data used")
print("   %s neural network design used" % args.neural_net)
print("   %2d filters used in the first CNN layer of the neural network" % args.num_filter)
print(" ")
print("   prediction lasted %4.3f seconds" % inference_time)
print(" ")

##
## Output a visual comparison between the two autoencoder designs and the CRS output
##

crs_output = np.load( "../input/crs_test.npy" )
plot_images( crs_output[:5,:,:], output[:5,:,:], args.neural_net, args.num_filter )

##
## Perform a pixel-by-pixel accuracy check. Determine # of pixels with less than 5% difference
## between the CRS Model and autoencoder outputs
##

print("   PREDICTION ACCURACY")

output = np.squeeze(output )
tmp = np.absolute( crs_output - output )
tol = 0.05

for n in range(3):
    cnt = (tmp < tol).sum()
    accuracy = (cnt / 1600.0) / np.float(crs_output.shape[0]) 
    print("   within %2.0f percent tolerance - %5.2f" % (tol*100.0,accuracy))
    tol = tol * 2.0
print(" ")

