import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import globalFunctions


# We start by creating placeholders for the data and the labels
n_input = 1600
n_output = 51

x = tf.placeholder(tf.float32, [None, 40, 40, 1])
y = tf.placeholder(tf.float32, [None, n_output])
phase_train = tf.placeholder(tf.bool)

# define the input layer
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

# Weight matrix is [height x width x input_channels x output_channels]
# Bias is [output_channels]
filter_size = 3
n_filters_1 = 16
n_filters_2 = 32
n_filters_3 = 64

# parameters
W_conv1 = globalFunctions.weight_variable([filter_size, filter_size, 1, n_filters_1])
b_conv1 = globalFunctions.bias_variable([n_filters_1])
W_conv2 = globalFunctions.weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = globalFunctions.bias_variable([n_filters_2])
W_conv3 = globalFunctions.weight_variable([filter_size, filter_size, n_filters_2, n_filters_3])
b_conv3 = globalFunctions.bias_variable([n_filters_3])

# layers
h_conv1 = tf.nn.relu(globalFunctions.conv2d(x_tensor, W_conv1) + b_conv1)
h_pool1 = globalFunctions.max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(globalFunctions.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = globalFunctions.max_pool_2x2(h_conv2)

h_conv3 = tf.nn.relu(globalFunctions.conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = globalFunctions.max_pool_2x2(h_conv3)

# 5x5 is the size of the image after the convolutional and pooling layers (40x40 -> 20x20 -> 10x10 -> 5x5)
h_conv3_flat = tf.reshape(h_pool3, [-1, 5 * 5 * n_filters_3])


# %% Create the one fully-connected layer:
n_fc = 1200
W_fc1 = globalFunctions.weight_variable([5 * 5 * n_filters_3, n_fc])
b_fc1 = globalFunctions.bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

n_fc2 = 800
W_fc2 = globalFunctions.weight_variable([n_fc, n_fc2])
b_fc2 = globalFunctions.bias_variable([n_fc2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = globalFunctions.weight_variable([n_fc2, n_output])
b_fc3 = globalFunctions.bias_variable([n_output])
y_pred = tf.matmul(h_fc1_drop2, W_fc3) + b_fc3

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

saver.restore(sess, "FinalDataset/Model/Model_Oron_1/Model")

# train_image,train_label,validation_image,validation_label,test_image,test_label =\
#     globalFunctions.load_from_file(0, 'TF', 40, True, force_load=False)
#
# prediction = tf.argmax(y_pred, 1)
# for i in range(0, len(test_image)):
#     test_new_image = []
#     test_new_image.append(test_image[i])
#     plt.imshow(np.resize(test_image[i].flatten(), [40,40]))
#     labels = prediction.eval(feed_dict={x: test_new_image, keep_prob: 1.0}, session=sess)
#     print("image {} is {}".format(i,labels[0]))


dir_images_list = listdir('FinalDataset/test_jpg')

hits_num = 0
images_num = 0

# For every label
for i in range(0, len(dir_images_list)):
    dir_images_list[i] = 'FinalDataset/test_jpg/' + dir_images_list[i]

for i in range(0, len(dir_images_list)):
    test_new_image = []
    test_new_label = []

    if dir_images_list[i][23] == '_':
        test_new_label.append(dir_images_list[i][22])
    else:
        test_new_label.append(dir_images_list[i][22:24])

    list_new = []
    list_new.append(dir_images_list[i])
    filename_queue = tf.train.string_input_producer(list_new)  # list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    channels = 1

    my_img = tf.image.decode_jpeg(value, channels=channels)  # use png or jpg decoder based on your files.
    my_img_re = tf.image.resize_images(my_img, [40, 40])

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess1:
        sess1.run(init_op)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = my_img_re.eval()  # here is your image Tensor :)
        resized_image = np.asarray(image)
        resized_image = resized_image / 255.

        test_new_image.append(resized_image)

        coord.request_stop()
        coord.join(threads)

    # list to numpy array.
    test_new_image = np.array(test_new_image)

    prediction = tf.argmax(y_pred, 1)
    labels = prediction.eval(feed_dict={x: test_new_image, keep_prob: 1.0}, session=sess)
    if int(test_new_label[0]) == labels[0]:
        result = "success"
        hits_num = hits_num + 1
    else:
        result = "error"
    images_num = images_num + 1

    print("image {} is {} - {}".format(dir_images_list[i][22:], labels[0], result))

accuracy_percent = hits_num * 100.0 / images_num
print("{} hits of {} images - {}% of accuracy".format(hits_num, images_num, accuracy_percent))