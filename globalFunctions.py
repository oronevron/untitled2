import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

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
            resized_image = cv2.resize(img, (40, 40))
            resized_image = resized_image.flatten()
            resized_image = resized_image / 255.

            image_list.append(resized_image)
            labal_hot.append(labal)

        # Shuffle data.
        random.shuffle(image_list)

        # Calc size of each group(train-validation-test)
        image_number = len(image_list)
        test_image_size = int(math.ceil(image_number * 0.8))
        validation_image_size = int(math.ceil(image_number * 0.9))

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

    # Shuffle data.
    train_image, train_labal = Shuffle(train_image, train_labal)
    validation_image, validation_labal = Shuffle(validation_image, validation_labal)
    test_image, test_labal = Shuffle(test_image, test_labal)

    # Save image.
    save_to_file(train_image,train_labal,validation_image,validation_labal,test_image,test_labal)

    return train_image,train_labal,validation_image,validation_labal,test_image,test_labal

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
        train_image, train_labal, validation_image, validation_labal, test_image, test_labal = load_image()

    # Return data.
    return train_image, train_labal, validation_image, validation_labal, test_image, test_labal

# Save image to binary file.
def save_to_file(train_image, train_labal, validation_image, validation_labal, test_image, test_labal):
      np.save('FinalDataset/bin/train_image.npy',train_image)
      np.save('FinalDataset/bin/train_labal.npy',train_labal)
      np.save('FinalDataset/bin/validation_image.npy',validation_image)
      np.save('FinalDataset/bin/validation_labal.npy', validation_labal)
      np.save('FinalDataset/bin/test_image.npy',test_image)
      np.save('FinalDataset/bin/test_labal.npy',test_labal)

def Shuffle(Image, Label):
    perm = np.arange(len(Image))
    np.random.shuffle(perm)
    Image = Image[perm]
    Label = Label[perm]

    return Image, Label
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# functions for parameter initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)