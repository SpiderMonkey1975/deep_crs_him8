from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
import numpy as np
import argparse, cnn

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--arch', type=str, default="volta", help="set the GPU architecture. Valid values are volta, pascal or kepler")
parser.add_argument('-p', '--padding', type=str, default="valid", help="set ipadding type for CNN. Valid values are valid or same")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-c', '--channels', type=int, default=3, help="number of channels in input data. Valid values are 3, 4, 5, 10")
parser.add_argument('-n', '--num_trials', type=int, default=5, help="number of trials for training. Valid value is between 1 and 5")
args = parser.parse_args()

if args.channels != 3 and args.channels != 4 and args.channels != 5 and args.channels != 10:
   args.channels = 3

if args.arch != "kepler" and args.arch != "pascal" and args.arch != "volta":
   args.arch = "volta"

if args.num_trials<1:
   args.num_trials = 1

if args.num_trials>5:
   args.num_trials = 5

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
## Set up the training of the model
##----------------------------------

filename = "model_weights/" + args.arch + "/weights_" + str(args.channels) + "channels_" + args.padding + "_padding.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=0.0005,
                           patience=10,
                           mode='min' )

history = History()

my_callbacks = [checkpoint, earlystop, history]

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")
print("   %s GPU architecture used" % args.arch)
print("   %2d channels of satellite data used" % args.channels)
print("   batch size of %3d images used" % args.batch_size)

for n in range(args.num_trials):

##
## Perform model training
##------------------------

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
   print("   TRAINING OUTPUT")
   print("       minimum val_loss was %8.6f" % min_val_loss)
   print("       minimum val_loss occurred at epoch %2d" % ind)
   print("       training lasted for %7.1f seconds" % training_time)

print(" ")
