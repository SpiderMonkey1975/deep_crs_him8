from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from plotting_routines import plot_images

import numpy as np
import argparse, neural_nets
from tiramisu_net import Tiramisu
from alt_model_checkpoint import AltModelCheckpoint

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--num_gpu', type=int, default=1, help="set number of GPUs to be used for training")
parser.add_argument('-f', '--num_filter', type=int, default=32, help="set initial number of filters used in CNN layers for the neural networks")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-n', '--neural_net', type=str, default='basic_autoencoder', help="set neural network design. Valid values are basic_autoencoder, unet and tiramisu")
args = parser.parse_args()

if args.neural_net!='basic_autoencoder' and args.neural_net!='unet' and args.neural_net!='tiramisu':
   args.neural_net = 'basic_autoencoder'

##
## Read in the input (X,Y) datasets
##

x = np.load( "../input/input_3layer_train.npy" )
y = np.load( "../input/crs_train.npy" )[:,:,:,None]

##
## Form the neural network
##

if args.neural_net == 'tiramisu':
   model = Tiramisu( input_shape=(400,400,3),
                     n_filters_first_conv=args.num_filter,
                     n_pool = 2,
                     n_layers_per_block = [4,5,7,5,4] ) 

if args.neural_net == 'basic_autoencoder':
    model = neural_nets.autoencoder( args.num_filter, args.num_gpu )

if args.neural_net == 'unet':
    model = neural_nets.unet( args.num_filter, args.num_gpu )
    if args.batch_size > 20:
       args.batch_size = 20

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

##
## Set up the training of the model
##

filename = "model_weights_" + args.neural_net + "_" + str(args.num_filter) + "filters.h5"
if args.num_gpu > 1:
   checkpoint = AltModelCheckpoint( filename, model,  
                                    monitor='val_loss', 
                                    save_best_only=True, 
                                    mode='min' )
else:
   checkpoint = ModelCheckpoint( filename, 
                                 monitor='val_loss', 
                                 save_best_only=True, 
                                 mode='min' )

earlystop = EarlyStopping( min_delta=0.0001,
                           patience=25,
                           mode='min' )

history = History()

my_callbacks = [checkpoint, earlystop, history]

##
## Perform model training
##

t1 = datetime.now()
hist = model.fit( x, y, 
                  batch_size=args.batch_size,
                  epochs=500, 
                  verbose=2, 
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
print("   %s neural network design used with %3d initial filters used in CNN layers" % (args.neural_net,args.num_filter))
print("   3 channels of satellite data used")
print("   batch size of %3d images used" % args.batch_size)
print(" ")
print("   TRAINING OUTPUT")
print("       minimum val_loss was %8.6f" % min_val_loss)
print("       minimum val_loss occurred at epoch %2d" % ind)
print("       training lasted for %7.1f seconds" % training_time)
print(" ")

