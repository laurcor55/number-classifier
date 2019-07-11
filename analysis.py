from mnist import MNIST
import cv2
import numpy as np
import matplotlib.pyplot as plt

data = MNIST('samples')
images, labels = data.load_training()

xdim = 28
ydim = 28
total_trainers = 1000
total_testers = 1000

def extract_image(sample_number, images, labels):
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
  
  return labels[sample_number], image_matrix

def create_filter(total_trainers):
  samples_per_number = np.zeros(10)

  average_filter = []
  for ii in range(10):
    average_filter.append(np.zeros((ydim, xdim)))

  for ii in range(total_trainers):
    sample_number = ii
    image_label, image_matrix = extract_image(sample_number, images, labels)
    average_filter[image_label] = np.add(average_filter[image_label], image_matrix)
    samples_per_number[image_label] += 1

  for ii in range(10):
    average_filter[ii] = np.divide(average_filter[ii], samples_per_number[ii])

  return average_filter

def test_accuracy(average_filter, total_trainers, total_testers):
  total_correct = 0

  for ii in range(total_trainers, total_trainers + total_testers):
    sample_number = ii
    
    image_label, image_matrix = extract_image(sample_number, images, labels)
    sample_correlation = np.zeros(10)
    for jj in range(10):
      sample_correlation[jj] = np.sum(np.sum(np.multiply(image_matrix, average_filter[jj])))
    guessed_number = np.argmax(sample_correlation)
    if (guessed_number==image_label):
      total_correct += 1
  return total_correct/total_testers*100

average_filter = create_filter(total_trainers)
for ii in range(9):
  plt.subplot(3, 3, ii+1)
  plt.imshow(average_filter[ii])
plt.show()

accuracy = test_accuracy(average_filter, total_trainers, total_testers)

print('accuracy:', accuracy, '%')