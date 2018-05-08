import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color, restoration
from os import listdir
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.morphology import *
from skimage.filters import frangi, rank, threshold_mean

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
image2 = image.copy()
for i in range(len(image)):
    for j in range(len(image[i])):
        if image[i][j] > 0.35:
            image[i][j] = lastpixel
        else:
            lastpixel = image[i][j]
filteredImageNotGaussian = image.copy()


filteredImageNotGaussian = img_as_ubyte(filteredImageNotGaussian)

filteredImageMorf = filteredImageNotGaussian.copy()
#for i in range(2):
filteredImageMorf = erosion(filteredImageMorf,np.ones((5,5),np.uint8))

'''
radius = 5
selem = disk(radius)

local_otsu = rank.otsu(filteredImageNotGaussian, selem)
binary = filteredImageNotGaussian < local_otsu
binary = img_as_ubyte(binary)
'''


binary = frangi(filteredImageMorf)
binary2 = frangi(filteredImageNotGaussian)

thresh = threshold_mean(binary)
binary = binary > thresh

fig, (ax0, ax1) = plt.subplots(nrows=2,
                                    ncols=2,
                                    sharex=True,
                                    sharey=True)


ax0[0].imshow(image2, cmap="gray")
ax0[1].imshow(filteredImageNotGaussian, cmap="gray")
ax1[0].imshow(binary, cmap="gray")
ax1[1].imshow(binary2, cmap="gray")

plt.show()

