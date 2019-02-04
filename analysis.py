from mnist import MNIST
import cv2
import numpy as np

data = MNIST('samples')
images, labels = data.load_training()
  

def extractImage(sampleNumber, images, labels):
  image = images[sampleNumber]
  imageString = data.display(images[sampleNumber])
  xdim = imageString.find('\n', 2, len(imageString))-1
  ydim = int(len(image)/xdim)

  imageMatrix = np.zeros((ydim, xdim))

  index = 0
  for ii in range(ydim):
    for jj in range(xdim):
      imageMatrix[ii, jj] = image[index]
      index += 1
  cv2.imshow('figure', imageMatrix)
  cv2.waitKey(10)
  print(labels[sampleNumber])


for ii in range(100):
  sampleNumber = ii
  extractImage(sampleNumber, images, labels)

