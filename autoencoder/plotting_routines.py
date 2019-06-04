import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_colormap():

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
    return newcmp

def plot_images( real_images, fake_images, net_type ):
    newcmp = create_colormap()
    
    filename = 'rainfall_3channels_regression_' + net_type + '.png'
    img_cols = real_images.shape[1] 
    img_rows = real_images.shape[2]
    plt.figure(figsize=(12,4))
    for i in range(real_images.shape[0]):
            plt.subplot(2, 5, i+1)
            image = real_images[ i,:,: ]
            image = np.reshape(image, [img_rows,img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
            if i == 0:
               plt.text( 0, 0, 'CRS Model Output', fontsize=14 )
            if i == 4:
               plt.colorbar()

    for i in range(fake_images.shape[0]):
            plt.subplot(2, 5, i+6)
            image = fake_images[ i,:,: ]
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
            if i == 0:
                if net_type == 'basic':
                   plt.text( 0, 0, 'Basic Autoencoder Output', fontsize=14 )
                else:
                   plt.text( 0, 0, 'U-Net Autoencoder Output', fontsize=14 )
            if i == 4:
               plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')

def compare_images( crs_output, basic_output, unet_output ):
    newcmp = create_colormap()

    filename = 'rainfall_3channels_regression_comparison.png'
    img_cols = crs_output.shape[1]
    img_rows = crs_output.shape[2]
    plt.figure(figsize=(12,6))
    for i in range(crs_output.shape[0]):
            plt.subplot(3, 5, i+1)
            image = crs_output[ i,:,: ]
            image = np.reshape(image, [img_rows,img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
            if i == 0:
               plt.text( 0, 0, 'CRS Model Output', fontsize=14 )
            if i == 4:
               plt.colorbar()

    for i in range(basic_output.shape[0]):
            plt.subplot(3, 5, i+6)
            image = basic_output[ i,:,: ]
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
            if i == 0:
               plt.text( 0, 0, 'Basic Autoencoder Output', fontsize=14 )
            if i == 4:
               plt.colorbar()

    for i in range(unet_output.shape[0]):
            plt.subplot(3, 5, i+11)
            image = unet_output[ i,:,: ]
            image = np.reshape(image, [img_rows, img_cols])
            plt.imshow(image, cmap=newcmp)
            plt.axis('off')
            if i == 0:
               plt.text( 0, 0, 'U-Net Autoencoder Output', fontsize=14 )
            if i == 4:
               plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')