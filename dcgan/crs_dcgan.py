'''
DCGAN on CRS model and Himawari8 satellite data using Keras
Author: Mark Cheeseman
Project: https://github.com/SpiderMonkey1975/deep-crs-him8
Dependencies: tensorflow 1.13 and keras 2.0
Usage: python3 train.py
'''

import numpy as np
import time,random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape, InputLayer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=400, img_cols=400, channel=3):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 32 
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, 1)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 3, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 3, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
    #    self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 128 

        input_shape = (self.img_rows, self.img_cols, 1)
        self.G.add(Conv2D(32, 5, strides=2, input_shape=input_shape, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        #self.G.add(Dropout(dropout))

        self.G.add(Conv2D(64, 3, strides=2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2D(128, 3, strides=2, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2D(256, 5, strides=1, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 400 x 400 x 1 image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
#        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class CRS_DCGAN(object):
    def __init__(self):
        self.img_rows = 400
        self.img_cols = 400
        self.channel = 3 

        self.x_train = np.load( "../input/crs.npy" )
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, batch_size=256):
        filename = "../input/input_" + str(self.channel) + "layer.npy"
        satellite_input = np.load( filename )

        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        #y2 = np.ones([batch_size, 1])

        for channel_no in range(self.channel):
            indicies = np.arange(self.x_train.shape[0])
            np.random.shuffle( indicies )
            print("channel %1d processing..." % (channel_no)) 
            cnt = 0
            for i in range(0,self.x_train.shape[0],batch_size):
                if i+batch_size > self.x_train.shape[0]:
                   break 

                ind = indicies[ i:i+batch_size ]
                #images_train = self.x_train[ind, :, :, :]
                reflectance_data = np.expand_dims( satellite_input[ ind,:,:,channel_no ], axis=3 )
                images_fake = self.generator.predict( reflectance_data )
                x = np.concatenate((self.x_train[ind,:,:,:], images_fake))
                d_loss = self.discriminator.train_on_batch( x, y )
                a_loss = self.adversarial.train_on_batch( reflectance_data, y[:batch_size] )
                if cnt == 10:
                   log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
                   log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                   print(log_mesg)
                   cnt = 0
                cnt = cnt + 1
            self.plot_images( channel_no=channel_no )

    def plot_images(self, samples=16, channel_no=0):   
        i = np.random.randint(0, self.x_train.shape[0], samples)
        
        fake_filename = 'generated_rainfall_channel' + str(channel_no) + '.png'
        input_file = "../input/input_" + str(self.channel) + "layer.npy"
        satellite_input = np.load( input_file )
        #channel_no = np.random.randint(0,self.channel, size=1)
        reflectance_data = np.expand_dims( satellite_input[ i,:,:,channel_no ], axis=3 )
        fake_images = self.generator.predict(reflectance_data)
        
        filename = 'true_rainfall_channel' + str(channel_no) + '.png'
        images = self.x_train[i, :, :, :]

        red = np.array([255, 252, 250, 247, 244, 242, 239, 236, 234, 231, 229, 226, 223, 221, 218, 215, 213, 210,
                     207, 205, 202, 199, 197, 194, 191, 189, 186, 183, 181, 178, 176, 173, 170, 168, 165, 162,
                     157, 155, 152, 150, 148, 146, 143, 141, 139, 136, 134, 132, 129, 127, 125, 123, 120, 118,
                     116, 113, 111, 109, 106, 104, 102, 100, 97,  95,  93,  90,  88,  86,  83,  81,  79,  77,
                     72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,  72,
                     72,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,  73,
                     73,  78,  83,  87,  92,  97,  102, 106, 111, 116, 121, 126, 130, 135, 140, 145, 150, 154,
                     159, 164, 169, 173, 178, 183, 188, 193, 197, 202, 207, 212, 217, 221, 226, 231, 236, 240,
                     245, 250, 250, 250, 250, 249, 249, 249, 249, 249, 249, 249, 249, 248, 248, 248, 248, 248,
                     248, 248, 247, 247, 247, 247, 247, 247, 247, 246, 246, 246, 246, 246, 246, 246, 246, 245,
                     245, 245, 244, 243, 242, 241, 240, 239, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230,
                     229, 228, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 217, 216, 215, 214,
                     213, 211, 209, 207, 206, 204, 202, 200, 199, 197, 195, 193, 192, 190, 188, 186, 185, 183,
                     181, 179, 178, 176, 174, 172, 171, 169, 167, 165, 164, 162, 160, 158, 157, 155, 153, 151, 150, 146], dtype = np.float)

        red = red / 255
        green = np.array([255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 243, 242, 241, 240, 239, 238,
                     237, 236, 235, 234, 233, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220,
                     218, 216, 214, 212, 210, 208, 206, 204, 202, 200, 197, 195, 193, 191, 189, 187, 185, 183,
                     181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 160, 158, 156, 154, 152, 150, 148, 146,
                     142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160,
                     161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179,
                     181, 182, 184, 185, 187, 188, 189, 191, 192, 193, 195, 196, 198, 199, 200, 202, 203, 204,
                     206, 207, 209, 210, 211, 213, 214, 215, 217, 218, 220, 221, 222, 224, 225, 226, 228, 229,
                     231, 232, 229, 225, 222, 218, 215, 212, 208, 205, 201, 198, 195, 191, 188, 184, 181, 178,
                     174, 171, 167, 164, 160, 157, 154, 150, 147, 143, 140, 137, 133, 130, 126, 123, 120, 116,
                     113, 106, 104, 102, 100,  98,  96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,
                     72,  70,  67,  65,  63,  61,  59,  57,  55,  53,  51,  49,  47,  45,  43,  41,  39,  37,
                     35,  31,  31,  30,  30,  30,  30,  29,  29,  29,  29,  28,  28,  28,  27,  27,  27,  27,
                     26,  26,  26,  26,  25,  25,  25,  25,  24,  24,  24,  23,  23,  23,  23,  22,  22,  22, 22,  21], dtype = np.float)

        green = green / 255
        blue = np.array([255, 255, 255, 254, 254, 254, 254, 253, 253, 253, 253, 253, 252, 252, 252, 252, 252, 251,
                     251, 251, 251, 250, 250, 250, 250, 250, 249, 249, 249, 249, 249, 248, 248, 248, 248, 247,
                     247, 246, 245, 243, 242, 241, 240, 238, 237, 236, 235, 234, 232, 231, 230, 229, 228, 226,
                     225, 224, 223, 221, 220, 219, 218, 217, 215, 214, 213, 212, 211, 209, 208, 207, 206, 204,
                     202, 198, 195, 191, 188, 184, 181, 177, 173, 170, 166, 163, 159, 156, 152, 148, 145, 141,
                     138, 134, 131, 127, 124, 120, 116, 113, 109, 106, 102, 99,  95,  91,  88,  84,  81,  77,
                     70,  71,  71,  72,  72,  73,  74,  74,  75,  75,  76,  77,  77,  78,  78,  79,  80,  80,
                     81,  81,  82,  82,  83,  84,  84,  85,  85,  86,  87,  87,  88,  88,  89,  90,  90,  91,
                     91,  92,  91,  89,  88,  86,  85,  84,  82,  81,  80,  78,  77,  75,  74,  73,  71,  70,
                     69,  67,  66,  64,  63,  62,  60,  59,  58,  56,  55,  53,  52,  51,  49,  48,  47,  45,
                     44,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,  41,
                     41,  41,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,  40,
                     40,  40,  40,  39,  39,  38,  38,  38,  37,  37,  36,  36,  36,  35,  35,  34,  34,  34,
                     33,  33,  32,  32,  31,  31,  31,  30,  30,  29,  29,  29,  28,  28,  27,  27,  27,  26, 26,  25], dtype = np.float)

        blue = blue / 255
        vals = np.ones((254, 4))
        vals[:, 0] = red
        vals[:, 1] = green
        vals[:, 2] = blue
        newcmp = ListedColormap(vals)

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')

        plt.figure(figsize=(10,10))
        for i in range(fake_images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = fake_images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(fake_filename)
        plt.close('all')
        
if __name__ == '__main__':
    crs_dcgan = CRS_DCGAN()
    timer = ElapsedTimer()
    crs_dcgan.train(batch_size=64)
    timer.elapsed_time()
    #crs_dcgan.plot_images()
