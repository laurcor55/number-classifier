from mnist import MNIST
import cv2 
import numpy as np

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

def normal_positive_image(image_matrix):
  minValue = np.min(image_matrix)
  normal_image_matrix = np.subtract(image_matrix, minValue*2)
  maxValue = np.max(normal_image_matrix)
  normal_image_matrix = np.divide(normal_image_matrix, maxValue)
  return normal_image_matrix

def convolution(image_matrix, filter):
  filter_size = filter.shape[0]
  correlation = np.zeros((image_matrix.shape[0]+filter_size, image_matrix.shape[1]+filter_size))
  image_matrix_padded = cv2.copyMakeBorder(image_matrix, filter_size, filter_size, filter_size, filter_size, cv2.BORDER_CONSTANT, value=0)

  for ii in range(ydim+filter_size):
    for jj in range(xdim+filter_size):
      sample = image_matrix_padded[ii:ii+filter_size, jj:jj+filter_size]
      correlation[ii, jj] = abs(np.sum(np.sum(np.multiply(filter, sample))))
  correlation = (correlation > 2*np.max(correlation)/3) * np.ones((correlation.shape[0], correlation.shape[1]))
  return correlation

average_filter = create_filter(total_trainers)
accuracy = test_accuracy(average_filter, total_trainers, total_testers)

print('accuracy:', accuracy, '%')

filter_size = 3
edge_detector_h = np.zeros((filter_size, filter_size))
edge_detector_h[0,:] = -0.5
edge_detector_h[1,:] = 1
edge_detector_h[2,:] = -0.5

edge_detector_v = np.zeros((filter_size, filter_size))
edge_detector_v[:,0] = -0.5
edge_detector_v[:,1] = 1
edge_detector_v[:,2] = -0.5

edge_detector_d = np.zeros((filter_size, filter_size))
edge_detector_d[0,0] = 1
edge_detector_d[1,1] = 1
edge_detector_d[2,2] = 1
edge_detector_d[0,1] = -.5
edge_detector_d[1,2] = -.5
edge_detector_d[1,0] = -.5
edge_detector_d[2,1] = -.5

label, image_matrix = extract_image(5, images, labels)
correlation_h = convolution(image_matrix, edge_detector_h)
correlation_v = convolution(image_matrix, edge_detector_v)
correlation_d = convolution(image_matrix, edge_detector_d)

cv2.imshow('figure1', image_matrix)
cv2.imshow('figure2', correlation_h)
cv2.imshow('figure3', correlation_v)
cv2.imshow('figure4', correlation_d)
cv2.waitKey(0)