import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse 
from datetime import datetime 

from my_utils import plot_images

print(" ")
print(" ")
print("=====================================================================================")
print("                          Rainfall Regression Network")
print("=====================================================================================")
print(" ")

##
## Look for any user specified commandline arguments
##

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=32, help="set batch size to the GPU")
parser.add_argument('-e', '--epoch', type=int, default=2, help="set number of epoch")
parser.add_argument('-v', '--verbose', type=int, default=0, help="idisplay debugging/verbose info")
args = parser.parse_args()

##
## Read in the input (X,Y) datasets, Convert input data sets to [N, C, H, W] dimensions layout
##

satellite_input = np.swapaxes( np.load( "../input/input_3layer_train.npy" ),1,3 )
crs_model_output = np.expand_dims( np.load( "../input/crs_train.npy" ), axis=1 )

training_dataset = torch.utils.data.TensorDataset( torch.FloatTensor(satellite_input),
                                                   torch.FloatTensor(crs_model_output) )
training_dataloader = torch.utils.data.DataLoader( training_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True,
                                                   num_workers=2 )
if args.verbose==1:
    print( "   Input training dataset constructed" )

##
## Use GPU for training if available
##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.verbose==1:
    print( "   training to be done on %s" % device )

##
## Define the autoencoder neural network design and load onto GPU 
##

autoencoder = torch.nn.Sequential(
                torch.nn.BatchNorm2d(3),
                torch.nn.Conv2d(3,32,5,stride=2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32,64,3,stride=2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
                torch.nn.Conv2d(64,128,3,stride=2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128,256,5,stride=2),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(256),
                torch.nn.ConvTranspose2d(256,128,5,stride=2,output_padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(128),
                torch.nn.ConvTranspose2d(128,64,3,stride=2,output_padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(64),
                torch.nn.ConvTranspose2d(64,32,3,stride=2,output_padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.ConvTranspose2d(32,1,5,stride=2,output_padding=1),
              ).to(device)

##
## Set the loss function and optimizer to be used
##

criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

print("   3 channels of satellite reflectance data used")
print("   batch size of %3d images used" % args.batch_size)
print(" ")

t1 = datetime.now()
for epoch in range(args.epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(training_dataloader, 0):
        # get the inputs
        reflectance_data, crs_data = data
        reflectance_data, crs_data = reflectance_data.to(device), crs_data.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = autoencoder( reflectance_data )
        loss = criterion(outputs, crs_data)
        loss.backward()
        optimizer.step()

        # print statistics
        if args.verbose==1:
           running_loss += loss.item()
           if i % 10 == 9:    # print every 10 mini-batches
               print('[Epoch %d] loss: %.3f' % (epoch+1, running_loss / 10))
               running_loss = 0.0

training_time = (datetime.now()-t1 ).total_seconds()

print(" ")
print("   %2d epochs performed in %7.1f seconds" % (args.epoch,training_time))

##
## Test the trained model
##

tmp = np.swapaxes( np.load( "../input/input_3layer_test.npy" ),1,3 )
satellite_input = torch.FloatTensor( tmp )

tmp2 = np.expand_dims( np.load( "../input/crs_test.npy" ), axis=1 )

if args.verbose==1:
   mse = tmp2*tmp2
   print("   max MSE zero-base error is ", np.sqrt(np.amax(mse)))
   mse = tmp2 - tmp 
   print("   max MSE difference-base error is ", np.sqrt(np.amax(mse)))

crs_model_output = torch.FloatTensor( tmp2 )
reflectance_data, crs_data = reflectance_data.to(device), crs_data.to(device)

if args.verbose==1:
    print(" ")
    print( "   Input test dataset constructed" )

output = autoencoder( reflectance_data )
loss = criterion(output, crs_data)

print("   Test mean square error was %.3f" % loss.item())

##
## Output some graphical comparisons
##

tmp = crs_model_output.to('cpu')
real_images = tmp.detach().numpy()
real_images = real_images[ :5,0,:,: ]

tmp = output.to('cpu')
fake_images = tmp.detach().numpy()
fake_images = fake_images[ :5,0,:,: ]

plot_images( 3, real_images, fake_images )

