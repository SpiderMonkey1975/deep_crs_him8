import sys, argparse
import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

sys.path.insert(0, '../neural_network_architecture/')
from basic_autoencoder import autoencoder, construct_model
from unet import unet
from fc_densenet import Tiramisu
from simple_classifier import two_layer_classifier

sys.path.insert(0, '../plotting_routines')
from plotting_routines import plot_images

##
## Perform the training of the GANs
##

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=250, help="set number of epochs performed in training")
parser.add_argument('-b', '--batch_size', type=int, default=100, help="set batch size to the GPU")
parser.add_argument('-f', '--num_filters', type=int, default=8, help="set number of filters used in first CNN layer of generator network")
parser.add_argument('-l', '--num_layers', type=int, default=3, help="set number of layers in the encoding (and decoding) part of the generator network")
parser.add_argument('-g', '--num_gpus', type=int, default=1, help="set number of GPUs used for training the GAN")
parser.add_argument('-w', '--weights_file', type=str, default='none', help="set name of model weights file for pretrained generator")
parser.add_argument('-n', '--neural_net', type=str, default='encoder-decoder', help="set architecture of generator network")
parser.add_argument('-o', '--output_frequency', type=int, default=25, help="set frequency in epochs at which output plots and model save files are generated")
args = parser.parse_args()

if args.num_layers>4:
   args.num_layers = 4

##
## Get the required input training and test data.
##

satellite_test_input = np.load( "../input/input_3layer_test.npy" )
satellite_test_input = satellite_test_input[ :5,:,:,: ]

real_test_images = np.load( "../input/crs_test.npy" )
real_test_images = real_test_images[:5,:,:]

X_train = np.load( "../input/crs_train.npy" )
Y_train = np.load( "../input/input_3layer_train.npy" )

batch_count = int(X_train.shape[0] / args.batch_size)
if batch_count*args.batch_size > X_train.shape[0]:
   batch_count = batch_count - 1

##
## Construct the discriminator network
##

discriminator = two_layer_classifier( args.num_filters )
discriminator.compile( loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'] )
discriminator.trainable = False

##
## Construct the generator network
##

if args.neural_net == 'encoder-decoder':
     generator = autoencoder( args.num_filters, args.num_gpus, args.num_layers )
elif args.neural_net == 'unet':
     generator = unet( args.num_filters, args.num_gpus )
elif args.neural_net == 'tiramisu':
     generator = Tiramisu( input_shape=(400,400,3), n_filters_first_conv=args.num_filters, n_pool = 2, n_layers_per_block = [4,5,7,5,4] )

generator.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])
if args.weights_file != 'none':
   generator.load_weights( args.weights_file )

##
## Construct the GAN network
##

gan_input = Input(shape = (400,400,3,))
x = generator(gan_input)

gan = construct_model( gan_input, discriminator(x), 1 )
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001))

##
## Perform the training
##

for e in tqdm(range(1,args.epoch+1 )):
    for n in range(batch_count):
        i1 = n*args.batch_size
        i2 = i1 + args.batch_size

        # Generate fake MNIST images from Himawari reflectance input data
        generated_images = generator.predict( Y_train[ i1:i2,:,:,: ] )

        # Get a set of images from the reference CRS model output data
        real_images = np.expand_dims( X_train[i1:i2,:,:], axis=3 )

        # Construct different batches of real and fake data
        X = np.concatenate([real_images, generated_images])

        # Labels for generated and real data
        y_dis=np.zeros(2*args.batch_size)
        y_dis[:args.batch_size]=0.9

        # Pre train discriminator on fake and real data  before starting the gan.
        discriminator.trainable=True
        discriminator.train_on_batch(X, y_dis)

        # Tricking the noised input of the Generator as real data
        y_gen = np.ones(args.batch_size)

        # During the training of gan,
        # the weights of discriminator should be fixed.
        # We can enforce that by setting the trainable flag
        discriminator.trainable=False

        # Train the GAN by alternating the training of the Discriminator
        # and training the chained GAN model with Discriminator’s weights freezed.
        gan.train_on_batch( Y_train[ i1:i2,:,:,: ], y_gen)

    if e % args.output_frequency == 0:
       fake_images = generator.predict( satellite_test_input )
       plot_images( real_test_images, fake_images, args.num_filters, e )
       weight_file = 'generator_weights_' + str(e) + 'epoch.h5'
       generator.save_weights( weight_file )

fake_images = generator.predict( satellite_test_input )
plot_images( real_test_images, fake_images, args.num_filters, e )

weight_file = 'generator_weights_' + str(e) + 'epoch.h5'
generator.save_weights( weight_file )
