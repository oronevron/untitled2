import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random
from os import listdir
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def load_image(image_type, load_type, size, with_rot):
    train_image = []
    train_labal = []
    validation_image = []
    validation_labal = []
    test_image = []
    test_labal = []

    dir_list = listdir('FinalDataset/jpg')

    # For every label
    for i in range(0,len(dir_list)):

        image_list = []
        labal_hot = []

        # Init lavel one hot vector
        labal = np.zeros(51)
        labal[int(dir_list[i])] = 1

        dir_images_list = listdir('FinalDataset/jpg/' + dir_list[i])

        if load_type == 'CV':
            # For every image
            for j in range(0,len(dir_images_list)):
                # Read image.
                img = cv2.imread('FinalDataset/jpg/' + str(dir_list[i]) + '/' + str(dir_images_list[j]),image_type)
                if (img is None):
                    continue

                # Resize image.
                resized_image = cv2.resize(img, (size, size))
                #resized_image = resized_image.flatten()
                resized_image = resized_image / 255.

                image_list.append(resized_image)
                labal_hot.append(labal)
        else:
            for j in range(0, len(dir_images_list)):
                dir_images_list[j] = 'FinalDataset/jpg/' + str(dir_list[i]) + '/' + dir_images_list[j]

            filename_queue = tf.train.string_input_producer(dir_images_list)  # list of files to read

            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)

            if image_type == 0:
                channels = 1
            else:
                channels = 3

            my_img = tf.image.decode_jpeg(value, channels=channels)  # use png or jpg decoder based on your files.
            my_img_re = tf.image.resize_images(my_img, [size, size])

            # Rotate iamge 180 degree
            my_img_re2 = tf.image.rot90(my_img_re, 2, name=None)

            init_op = tf.initialize_all_variables()

            with tf.Session() as sess:
                sess.run(init_op)

                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for m in range(0, len(dir_images_list)):  # length of your filename list
                    # print(str(i) + "," + file_list[m])
                    image = my_img_re.eval()  # here is your image Tensor :)
                    resized_image = np.asarray(image)
                    # resized_image = resized_image.transpose(2,0,1).reshape(3,-1)
                    # resized_image = resized_image.transpose(1,0)
                    # resized_image = resized_image.squeeze()
                    # resized_image = resized_image.flatten()
                    resized_image = resized_image / 255.

                    image_list.append(resized_image)
                    labal_hot.append(labal)

                    if with_rot == True:
                        image = my_img_re2.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        image_list.append(resized_image)
                        labal_hot.append(labal)

                coord.request_stop()
                coord.join(threads)

        # Shuffle data.
        #random.shuffle(image_list)

        # Calc size of each group(train-validation-test)
        image_number = len(image_list)
        test_image_size = int(math.ceil(image_number * 0.84))
        validation_image_size = int(math.ceil(image_number * 0.92))

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
def load_from_file(image_type, load_type, size, with_rot,force_load=False):

    if force_load == False and os.path.exists('FinalDataset/bin/train_image.npy') is True:
        train_image = np.load('FinalDataset/bin/train_image.npy')
        train_labal = np.load('FinalDataset/bin/train_labal.npy')
        validation_image = np.load('FinalDataset/bin/validation_image.npy')
        validation_labal = np.load('FinalDataset/bin/validation_labal.npy')
        test_image = np.load('FinalDataset/bin/test_image.npy')
        test_labal = np.load('FinalDataset/bin/test_labal.npy')
    else:
        train_image, train_labal, validation_image, validation_labal, test_image, test_labal \
            = load_image(image_type, load_type, size, with_rot)

    # Return data.
    return train_image, train_labal, validation_image, validation_labal, test_image, test_labal
# def load_image():
#     train_image = []
#     train_labal = []
#     validation_image = []
#     validation_labal = []
#     test_image = []
#     test_labal = []
#
#     # For every label
#     for i in range(0,50):
#
#         image_list = []
#         labal_hot = []
#
#         # Init lavel one hot vector
#         labal = np.zeros(50)
#         labal[i] = 1
#
#         # For every image
#         for j in range(0,1000):
#             # Read image.
#             img = cv2.imread('FinalDataset/jpg/' + str(i) + '/' + str(j) + '.jpg', 0)
#             if (img is None):
#                 continue
#
#             # Resize image.
#             resized_image = cv2.resize(img, (40, 40))
#             resized_image = resized_image.flatten()
#             resized_image = resized_image / 255.
#
#             image_list.append(resized_image)
#             labal_hot.append(labal)
#
#         # Shuffle data.
#         random.shuffle(image_list)
#
#         # Calc size of each group(train-validation-test)
#         image_number = len(image_list)
#         test_image_size = int(math.ceil(image_number * 0.8))
#         validation_image_size = int(math.ceil(image_number * 0.9))
#
#         # Split current label image to train-validation-test
#         train_image.extend(image_list[:test_image_size])
#         train_labal.extend(labal_hot[:test_image_size])
#         validation_image.extend(image_list[test_image_size:validation_image_size])
#         validation_labal.extend(labal_hot[test_image_size:validation_image_size])
#         test_image.extend(image_list[validation_image_size:])
#         test_labal.extend(labal_hot[validation_image_size:])
#
#     # list to numpy array.
#     train_image = np.array(train_image)
#     train_labal = np.array(train_labal)
#     validation_image = np.array(validation_image)
#     validation_labal = np.array(validation_labal)
#     test_image = np.array(test_image)
#     test_labal = np.array(test_labal)
#
#     # Shuffle data.
#     train_image, train_labal = Shuffle(train_image, train_labal)
#     validation_image, validation_labal = Shuffle(validation_image, validation_labal)
#     test_image, test_labal = Shuffle(test_image, test_labal)
#
#     # Save image.
#     save_to_file(train_image,train_labal,validation_image,validation_labal,test_image,test_labal)
#
#     return train_image,train_labal,validation_image,validation_labal,test_image,test_labal
#
# # Load image from binary file.
# def load_from_file():
#
#     if os.path.exists('FinalDataset/bin/train_image.npy') is True:
#         train_image = np.load('FinalDataset/bin/train_image.npy')
#         train_labal = np.load('FinalDataset/bin/train_labal.npy')
#         validation_image = np.load('FinalDataset/bin/validation_image.npy')
#         validation_labal = np.load('FinalDataset/bin/validation_labal.npy')
#         test_image = np.load('FinalDataset/bin/test_image.npy')
#         test_labal = np.load('FinalDataset/bin/test_labal.npy')
#     else:
#         train_image, train_labal, validation_image, validation_labal, test_image, test_labal = load_image()
#
#     # Return data.
#     return train_image, train_labal, validation_image, validation_labal, test_image, test_labal

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

def one_hot_to_label_array(one_hot):
    label = []

    for i in range(0,len(one_hot)):
        label.append(np.where(one_hot[i]==1)[0][0])

    return label

def confusion_matrix1(actual, prediction):
    cm = confusion_matrix(actual,prediction)

    # df_cm = pd.DataFrame(cm, range(50),
    #                      range(50))
    # plt.figure(figsize = (50,40))
    # sn.set(font_scale=1.4)  # for label size
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 14})  # font size

def Restore_and_run_test(model_name):
    a = 1