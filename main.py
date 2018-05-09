import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color, restoration
from os import listdir
from skimage.util import img_as_ubyte
from skimage.morphology import *
from skimage.restoration import denoise_bilateral
from scipy.signal import fftconvolve
from skimage.filters import gabor_kernel, frangi, threshold_mean, gaussian, threshold_local
from skimage.filters.rank import mean
from skimage import feature

def generateGaborKernels():
    std = 10.0

    angles = [0, 30, 60, 90]
    freqs = [0.03, 0.05, 0.07, 0.09]

    scale = 1.0
    kernels = []

    for angle in angles:
        kernels_row = []
        num=0
        for freq in freqs:
            num += 1
            kernel = np.real(gabor_kernel(freq, theta=angle / 90.0 * 0.5 * np.pi, sigma_x=std, sigma_y=std))
            kernels_row.append(kernel)
        kernels.append(kernels_row)
    return kernels

def applyGaborsFilters(image, kernels):
    image = color.rgb2gray(image)
    sum_image = np.zeros(image.shape)

    for row in kernels:
        num = 0
        for kernel in row:
            num += 1
            img_convolve = fftconvolve(image, kernel, mode='same')
            # TODO
            # add img_convovle to sum_image
            sum_image += img_convolve
    averaged_image = sum_image / 16
    return averaged_image

def averageBrightSpots(image):
    m = np.mean(image)
    print(m)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 2*m:
                image[i][j] = 1.5*m
    return image

imagePath = "stare-images"

#----------------------------------------
filesList = listdir(imagePath)
for i in range(len(filesList)):
    filesList[i] = os.path.join(imagePath,filesList[i])

print(filesList)
originalImage = io.imread(filesList[0])
greenImage = originalImage.copy()
greenImage[:,:,0] = 0
greenImage[:,:,2] = 0


image = color.rgb2gray(greenImage)

#gaborKernels = generateGaborKernels()
#image = averageBrightSpots(image)




image = gaussian(image,sigma=3)
image = opening(image, square(5))


#image = averageBrightSpots(image)



#binary = black_tophat(image, square(5))
# = opening(image,square(5))
#binary = frangi(image)
binary = black_tophat(image,square(5))
thresh = threshold_local(binary, 21, offset=0.0004)
binary = binary < thresh


#binary = threshold(binary)
#binary = binary > thresh
#image = opening(image,square(15))

fig, (ax0, ax1) = plt.subplots(nrows=1,
                                    ncols=2,
                                    sharex=True,
                                    sharey=True)


ax0.imshow(image, cmap="gray")
ax1.imshow(binary, cmap="gray")


plt.show()

