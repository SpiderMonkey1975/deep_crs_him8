from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from my_utils import plot_images

import numpy as np
import argparse, cnn

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--padding', type=str, default="same", help="set padding type for CNN. Valid values are valid or same")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3 and 10")
args = parser.parse_args()

if args.channels != 3 and args.channels != 10:
   args.channels = 3

if args.padding != "valid" and args.padding != "same":
   args.padding = "same"

##
## Read in the input (X,Y) datasets
##

filename= "../input/input_" + str(args.channels) + "layer_train.npy"
x = np.load( filename )

y = np.load( "../input/crs_train.npy" )[:,:,:,None]

##
## Form the neural network
##

num_filters = 5
if args.padding == "valid":
   num_filters = 3
   y = y[:,:399,:399,:]

model = cnn.cnn( args.channels, args.padding, num_filters )

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

##
## Set up the training of the model
##

filename = "weights_" + str(args.channels) + "channels_" + args.padding + "_padding.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=0.0005,
                           patience=10,
                           mode='min' )

history = History()

my_callbacks = [checkpoint, earlystop, history]

##
## Perform model training
##

t1 = datetime.now()
hist = model.fit( x, y, 
                  batch_size=args.batch_size,
                  epochs=100, 
                  verbose=0, 
                  validation_split=.25,
                  callbacks=my_callbacks, 
                  shuffle=True )
training_time = (datetime.now()-t1 ).total_seconds()

val_loss = hist.history['val_loss']
min_val_loss = 10000000.0
ind = -1
for n in range(len(val_loss)):
       if val_loss[n] < min_val_loss:
          min_val_loss = val_loss[n]
          ind = n

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
if args.channels == 10:
   print("   %2d channels of satellite data used" % args.channels)
else:
   print("   %1d channels of satellite data used" % args.channels)
print("   batch size of %3d images used" % args.batch_size)
print(" ")
print("   TRAINING OUTPUT")
print("       minimum val_loss was %8.6f" % min_val_loss)
print("       minimum val_loss occurred at epoch %2d" % ind)
print("       training lasted for %7.1f seconds" % training_time)
print(" ")

##
## Perform model evaluation
##

filename = "../input/input_" + str(args.channels) + "layer_test.npy"
reflectance_data = np.load( filename )
real_images = np.load( "../input/crs_test.npy" )[:,:,:,None]

print("   EVALUATION OUTPUT")

t1 = datetime.now()
score = model.evaluate( reflectance_data, real_images, batch_size=args.batch_size, verbose=0 )
inference_time = (datetime.now()-t1).total_seconds()
print("       inference took %5.1f seconds" % inference_time )
print("       mean square error was %12.10f" % score[1] )

print(" ")
print(" ")

##
## Output some visual comparisons
##

idx = np.random.choice( real_images.shape[0], 5 ) 
real_images = real_images[ idx,:,: ]
fake_images = model.predict( reflectance_data[ idx,:,:,: ] )
plot_images( args.channels, real_images, fake_images )
