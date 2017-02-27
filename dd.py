import tensorflow as tf

#import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
import cv2
import math
import os.path
#import Image

import matplotlib.pyplot as plt

# read the data and labels as ont-hot vectors
# one-hot means a sparse vector for every observation where only
# the class label is 1, and every other class is 0.
# more info here:

def load_image():
      train_image = []
      train_labal = []
      validation_image = []
      validation_labal = []
      test_image = []
      test_labal = []

      # For every label
      for i in range(0,50):

            image_list = []
            labal_hot = []

            # Init lavel one hot vector
            labal = np.zeros(50)
            labal[i] = 1

            # For every image
            for j in range(0,1000):
                  # Read image.
                  img = cv2.imread('FinalDataset/jpg/' + str(i) + '/' + str(j) + '.jpg', 0)
                  if (img is None):
                        continue

                  # Resize image.
                  resized_image = cv2.resize(img, (50, 50))
                  resized_image = resized_image.flatten()
                  resized_image = resized_image / 255.

                  image_list.append(resized_image)
                  labal_hot.append(labal)

            # Split image to three group + create hot spor labal array.

            # Calc size of each group(train-validation-test)
            image_number = len(image_list)
            test_image_size = int(math.ceil(image_number * 0.7))
            validation_image_size = int(math.ceil(image_number * 0.85))

            # Split current label image to train-validation-test
            train_image.extend(image_list[:test_image_size])
            train_labal.extend(labal_hot[:test_image_size])
            validation_image.extend(image_list[test_image_size:validation_image_size])
            validation_labal.extend(labal_hot[test_image_size:validation_image_size])
            test_image.extend(image_list[validation_image_size:])
            test_labal.extend(labal_hot[validation_image_size:])

      # list to numpy array.
      train_image = np.array(train_image)
      train_labal = np.array(train_labal)
      validation_image = np.array(validation_image)
      validation_labal = np.array(validation_labal)
      test_image = np.array(test_image)
      test_labal = np.array(test_labal)

      # Save image.
      save_to_file(train_image,train_labal,validation_image,validation_labal,test_image,test_labal)

# Load image from binary file.
def load_from_file():

      if os.path.exists('FinalDataset/bin/train_image.npy') is True:
            train_image = np.load('FinalDataset/bin/train_image.npy')
            train_labal = np.load('FinalDataset/bin/train_labal.npy')
            validation_image = np.load('FinalDataset/bin/validation_image.npy')
            validation_labal = np.load('FinalDataset/bin/validation_labal.npy')
            test_image = np.load('FinalDataset/bin/test_image.npy')
            test_labal = np.load('FinalDataset/bin/test_labal.npy')
      else:
            load_image()

# Save image to binary file.
def save_to_file(train_image, train_labal, validation_image, validation_labal, test_image, test_labal):
      np.save('FinalDataset/bin/train_image.npy',train_image)
      np.save('FinalDataset/bin/train_labal.npy',train_labal)
      np.save('FinalDataset/bin/validation_image.npy',validation_image)
      np.save('FinalDataset/bin/validation_labal.npy', validation_labal)
      np.save('FinalDataset/bin/test_image.npy',test_image)
      np.save('FinalDataset/bin/test_labal.npy',test_labal)

# read data
#mnist1 = fetch_mldata("MNIST original")
#X, Y = mnist1.data[:1000] / 255., mnist1.target[:1000]

# Load an color image in grayscale
#ttt = np.load('data.npy')
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(28000)
end = len(batch_xs)
x = batch_xs[200:end]
batch_xs, batch_ys = mnist.train.next_batch(30000)
batch_xs, batch_ys = mnist.train.next_batch(30000)

batch_size = 100
n_epochs = 5
for epoch_i in range(n_epochs):
    for batch_i in range(0, mnist.train.num_examples, batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        print(batch_i)

abal = np.zeros(50)
load_from_file()
abal = np.zeros(50)
folder = 2
my_dict = {'list1': [], 'list2': []}
image = 0
img = cv2.imread(str(folder) + '/' + str(image) + '.jpg',0)
resized_image = cv2.resize(img, (50, 50))
resized_image = resized_image / 255.

#plt.imshow(resized_image, cmap='gray')
print type(resized_image)
aa = resized_image.flatten()

r = []
r.append(aa)
bb = resized_image.flatten()
r.append(bb)
r.append(bb)
r.append(bb)
bfffb = r[:2.0]
my_dict['list2']  = np.array(r)
np.save('data.npy', bla)
cc = np.reshape(aa, (-1,2500))
dd = np.reshape(bb, (-1,2500))
aa = np.vstack([aa, bb])
img = cv2.imread('blablea.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# mnist is now a DataSet with accessors for:
#'train', 'test', and 'validation'.
# within each, we can access:
# images, labels, and num_examples
print(mnist.train.num_examples,
      mnist.test.num_examples,
      mnist.validation.num_examples)

# the images are stored as:
# n_observations x n_features tensor (n-dim array)
# the labels are stored as n_observations x n_labels,
# where each observation is a one-hot vector.
print(mnist.train.images.shape, mnist.train.labels.shape)

# the range of the values of the images is from 0-1
print(np.min(mnist.train.images), np.max(mnist.train.images))

# we can visualize any one of the images by reshaping it to a 28x28 image
#plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')