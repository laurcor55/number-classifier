from mnist import MNIST
import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from collections import Counter

data = MNIST('samples')
images, labels = data.load_training()

xdim = 28
ydim = 28
images_train = images[0:1000]
labels_train = labels[0:1000]
images_test = images[1000:]
labels_test = labels[1000:]

def parse_image(sample_number, images):
  image = images[sample_number]
  image_string = data.display(images[sample_number])
  xdim = image_string.find('\n', 2, len(image_string))-1
  ydim = int(len(image)/xdim)
  image_matrix = np.zeros((ydim, xdim))
  index = 0
  for ii in range(ydim):
    for jj in range(xdim):
      image_matrix[ii, jj] = image[index]
      index += 1
  return image_matrix

total_correct = 0
for jj in range(100):
  image_test = parse_image(jj, images_test)
  label_test = labels_test[jj]
  fit = np.zeros(len(labels_train))
  for ii in range(len(labels_train)):
    image_train = parse_image(ii, images_train)
    fit[ii] = np.sum(np.abs(np.subtract(image_train, image_test)))
  fit_sort = np.argsort(fit)
  nearest_neighbors = 9
  fit_values = np.zeros(nearest_neighbors)
  for kk in range(nearest_neighbors):
    fit_values[kk] = labels_train[fit_sort[kk]]
  fit_value = Counter(fit_values).most_common()
  total_correct += (fit_value[0][0] == label_test)
print('Accuracy: ' + str(total_correct) + '%')