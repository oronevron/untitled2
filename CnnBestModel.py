import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

import globalFunctions

# In this CNN we have:
# first conv layer with 8 filters
# second conv layer with 16 filters
# third conv layer with 32 filters
# dense layer with 740 neurons
# keep prob of 0.4
# 10 epochs
# regular color mode (black and white)
# 40x40 image size
# the current data is improved by adding one random brightness and one random contrast to each image

# results are very high
# 95% - 96% accuracy
#globalFunctions.Image_split()

train_image,train_label,validation_image,validation_label,test_image,test_label = \
    globalFunctions.load_all_image(0, 'TF', 40, True, force_load=False, force_load_val=False)

count = 0

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
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables://
sess = tf.Session()
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

# We'll train in minibatches and report accuracy:
batch_size = 100
n_epochs = 100
l_loss = list()
l_loss_train = list()

for epoch_i in range(n_epochs):
    train_loss_sum = 0
    batch_counter = 0
    for batch_i in range(0, len(train_image), batch_size):
        # print('epoch {} barch: {}'.format(epoch_i + 1, batch_i))

        # Calc end of next batch
        end = batch_i + batch_size
        if end > len(train_image):
            end = len(train_image)
        batch_xs = train_image[batch_i:end]
        batch_ys = train_label[batch_i:end]
        opt, loss_train = sess.run([optimizer, accuracy], feed_dict={
            x: batch_xs, y: batch_ys, phase_train: True, keep_prob: 0.4})
        train_loss_sum = train_loss_sum + loss_train
        batch_counter = batch_counter + 1
    print('Train accuracy for epoch {} is: {}'.format(epoch_i + 1, (train_loss_sum / batch_counter)))
    l_loss_train.append(train_loss_sum / batch_counter)

    loss = sess.run(accuracy, feed_dict={
                       x: validation_image,
                       y: validation_label,
                       phase_train: False,
                       keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)

    # Shuffle data.
    train_image, train_label = globalFunctions.Shuffle(train_image, train_label)

    saver.save(sess, "FinalDataset/Model/aaa/" + str(epoch_i) + "/Model")
    globalFunctions.accuracy_per_epoch_validation(l_loss, "accuracy_per_epoch_validation",epoch_i)
    globalFunctions.accuracy_per_epoch_train_and_validation(l_loss_train, l_loss, "accuracy_per_epoch_train_and_validation", epoch_i )

saver.save(sess, "FinalDataset/Model/aaa/" + str(epoch_i) + "/Model")
print("Accuracy for test set: {}".format(sess.run(accuracy,
               feed_dict={
                   x: test_image,
                   y: test_label,
                   keep_prob: 1.0
               })))

