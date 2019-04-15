from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History, LearningRateScheduler
from datetime import datetime
import numpy as np
import argparse

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--arch', type=str, default="volta", help="set the GPU architecture. Valid values are volta, pascal or kepler")
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

def GetModel( num_channels ):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, num_channels)))
    # Size 400x400x3
    model.add(Conv2D(32, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2D(64, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2D(128, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2D(256, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 25x25x256
    model.add(Conv2DTranspose(128, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2DTranspose(64, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2DTranspose(32, 3, strides=2, activation='relu'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2DTranspose(1, 3, strides=2, activation='relu'))
    # Size 400x400x1

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
    return model


##
## Create a model
##----------------

model = GetModel( args.channels )


##
## Set up the training of the model
##----------------------------------

filename = "model_weights/" + args.arch + "/model_weights_cnn_" + str(args.channels) + "channels.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=0.001,
                           patience=10,
                           mode='min' )

history = History()

def step_decay(epoch):
   if epoch<21:
      lrate = 0.0001
   else:
      lrate = 0.00005
   return lrate

lrate = LearningRateScheduler(step_decay)

my_callbacks = [checkpoint, earlystop, history, lrate]

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
   hist = model.fit( x, y[:,:399,:399,:], 
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
