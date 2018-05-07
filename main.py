import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from os import listdir
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage.morphology import *
from skimage.filters import threshold_otsu, rank

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
for i in range(len(image)):
    for j in range(len(image[i])):
        if image[i][j] > 0.35:
            image[i][j] = lastpixel
        else:
            lastpixel = image[i][j]
#filteredImage = image.copy()
filteredImage = gaussian(image,sigma=3)
filteredImage = closing(filteredImage,np.ones((5,5),np.uint8))

filteredImage = img_as_ubyte(filteredImage)

radius = 5
selem = disk(radius)

local_otsu = rank.otsu(filteredImage, selem)
binary = filteredImage < local_otsu
binary = binary_closing(binary,np.ones((5,5),np.uint8))


fig, (ax0, ax1) = plt.subplots(nrows=1,
                                    ncols=2,
                                    sharex=True,
                                    sharey=True)
ax0.imshow(image, cmap="gray")
ax1.imshow(binary,cmap="gray")
plt.show()

