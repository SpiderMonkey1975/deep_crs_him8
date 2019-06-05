from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, History
from datetime import datetime
from plotting_routines import plot_images

import numpy as np
import argparse, neural_nets

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-n', '--neural_net', type=str, default='basic', help="set neural network design for autoencoder. Valid values are basic and unet")
args = parser.parse_args()

if args.neural_net!='basic' and args.neural_net!='unet':
    args.neural_net = 'basic'

##
## Read in the input (X,Y) datasets
##

x = np.load( "../input/input_3layer_train.npy" )
y = np.load( "../input/crs_train.npy" )[:,:,:,None]

##
## Form the neural network
##

if args.neural_net == 'basic':
    model = neural_nets.autoencoder()
else:
    model = neural_nets.unet()
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

##
## Set up the training of the model
##

filename = "weights_3channels_" + args.neural_net + "_autoencoder.h5"
checkpoint = ModelCheckpoint( filename, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='min' )

earlystop = EarlyStopping( min_delta=0.0001,
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
print("   %s autoencoder neural network design used" % args.neural_net)
print("   3 channels of satellite data used")
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

reflectance_data = np.load( "../input/input_3layer_test.npy" )
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
fake_images = model.predict( reflectance_data[ idx,:,:,: ] )
plot_images( real_images[ idx,:,: ], fake_images, args.neural_net )
