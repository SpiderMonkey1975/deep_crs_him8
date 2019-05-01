from tensorflow.keras.optimizers import Adam 
#from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import argparse, neural_networks, sys, random

verbose = False

##----------------
## Initial Setup
##----------------

# parse commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=8, help="set batch size to the GPU")
parser.add_argument('-n', '--num_epoch', type=int, default=1, help="set number of GANs epochs")
args = parser.parse_args()

# read in feature and label datasets 
input_dir = "../cnn/input/"

y_file = input_dir + "crs.npy"
x_file = input_dir + "input_3layer.npy"

x_train = np.load( x_file )[:,:,:,:]
y_true = np.load( y_file )[:,:,:,None]

generator = neural_networks.generator()
generator.compile( loss='mean_squared_error', 
                   optimizer=Adam(lr=0.0001), 
                   metrics=['mse'] )

discriminator = neural_networks.discriminator()
discriminator.compile( loss='binary_crossentropy', 
                       optimizer=Adam(lr=0.0001), 
                       metrics=['accuracy'] )

##--------------------------------------------
## Generate the initial "fake" label dataset
##--------------------------------------------

print(" ")
print("-----------------------------------------------")
print("    Generative Adversarial Networks Run")
print("-----------------------------------------------")
print(" ")
y_train = y_true

for epoch in range(args.num_epoch):

    # Train and run generator
    hist = generator.fit( x_train, y_train,
                          batch_size=args.batch_size,
                          epochs=10,
                          verbose=0,
                          validation_split=.25)
    feature_data = generator.predict( x_train, batch_size=args.batch_size, verbose=0 )

    if verbose:
       print("   generator output created")

    # one-hot encoding. 1 denotes a true image, 0 denotes a fake image
    true_ids = random.sample( range(y_true.shape[0]), int(0.5*y_true.shape[0]) )
    one_hot_encoding = np.zeros((y_true.shape[0],1), dtype=int)

#    one_hot_encoding[true_ids] = 1
#    feature_data[true_ids,:,:,:] = y_true[true_ids,:,:,:]
    for n in true_ids:
        one_hot_encoding[n] = 1
        feature_data[n,:,:,:] = y_true[n,:,:,:]

    if verbose:
       print("   one-hot encoding done")
 
    # Train and run the discriminator 
    hist = discriminator.fit( feature_data, one_hot_encoding, 
                              batch_size=args.batch_size, 
                              epochs=4, 
                              verbose=0,
                              validation_split=.25, 
                              shuffle=True )
    discriminator_guesses = discriminator.predict( feature_data, batch_size=args.batch_size, verbose=0 )

    if verbose:
       print("   discriminator testing done")

    num_true_positives = 0
    num_true_negatives = 0
    for n in range(feature_data.shape[0]):
        if one_hot_encoding[n] == 1 and discriminator_guesses[n] == 1:
           num_true_positives = num_true_positives + 1
        if one_hot_encoding[n] == 0 and discriminator_guesses[n] == 0:
           num_true_negatives = num_true_negatives + 1

    array_len = y_true.shape[0]-len(true_ids)
    acc = float(num_true_positives) / float(len(true_ids))
    acc2= float(num_true_negatives) / float(array_len)
    print(" Epoch %d: detection accuracies: %5.1f %5.1f" % (epoch,acc,acc2))
#    print("   # of true negatives was %3d" % (num_true_negatives))
#    print("   # of false positives was %3d" % (num_false_positives))


sys.exit()

##----------------------
## Train the generator 
##----------------------

hist = generator.fit( x, discriminator_guesses, 
                      batch_size=args.batch_size,
                      epochs=10,
                      validation_split=.25,
                      shuffle=True )

