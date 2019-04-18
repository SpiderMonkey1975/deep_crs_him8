from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import numpy as np
import argparse, cnn

##
## Look for any user specified commandline arguments
##---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--arch', type=str, default="pascal", help="set the GPU architecture. Valid values are volta, pascal or kepler")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3, 4, 5, 10")
parser.add_argument('-n', '--num_tries', type=int, default=5, help="number of inference tests. Minimum value of 5")
parser.add_argument('-p', '--padding', type=str, default="valid", help="set ipadding type for CNN. Valid values are valid or same")
args = parser.parse_args()

if args.channels != 3 and args.channels != 4 and args.channels != 5 and args.channels != 10:
   args.channels = 3

if args.num_tries < 5:
   args.num_tries = 5

if args.arch != "kepler" and args.arch != "pascal" and args.arch != "volta":
   args.arch = "pascal"

if args.padding != "valid" and args.padding != "same":
   args.padding = "same"

##
## Read in the input (X,Y) datasets
##----------------------------------
input_dir = "input/"

y_file = input_dir + "crs.npy"
x_file = input_dir + "input_" + str(args.channels) + "layer.npy"

x = np.load( x_file )
y = np.load( y_file )[:,:,:,None]

##
## Form the neural network
##-------------------------
num_filters = 5
if args.padding == "valid":
   num_filters = 3
   y = y[:,:399,:399,:]

model = cnn.cnn( args.channels, args.padding, num_filters )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

##
## Load pre-trained model weights
##--------------------------------
filename = "model_weights/" + args.arch + "/weights_" + str(args.channels) + "channels_" + args.padding + "_padding.h5"
model.load_weights( filename )

##
## Perform inference testing
##---------------------------
print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   %s GPU architecture used" % args.arch)
print("   %2d channels of satellite data used" % args.channels)
print("   batch size of %3d images used" % args.batch_size)
print(" ")
print("   INFERENCE OUTPUT: trial    time         MSE")

for n in range(args.num_tries):
    t1 = datetime.now()
    score = model.evaluate( x, y, batch_size=args.batch_size, verbose=0 )
    inference_time = (datetime.now()-t1).total_seconds()
    print("                         %1d       %5.1f        %12.10f" % (n,inference_time,score[1]) )
    
print(" ")
print(" ")

