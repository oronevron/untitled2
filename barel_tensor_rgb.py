import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random
from os import listdir


from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# load image with tensorflow as rgb image. and load the picture as tensor at the start the work.
# do 20 epoks - the best one was epok 9 with 77.0% accuracy. test data %77.6

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

        file_list = listdir('FinalDataset/jpg/' + str(i))
        if len(file_list) == 0:
            continue

        for j in range(0,len(file_list)):
            file_list[j] = 'FinalDataset/jpg/' + str(i) + '/' + file_list[j]

        filename_queue = tf.train.string_input_producer(file_list)  # list of files to read

        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)

        my_img = tf.image.decode_jpeg(value, channels=3)  # use png or jpg decoder based on your files.
        my_img2 = tf.image.resize_image_with_crop_or_pad(my_img, [40, 40])

        init_op = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init_op)

            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for m in range(0,len(file_list)):  # length of your filename list
                #print(str(i) + "," + file_list[m])
                image = my_img2.eval()  # here is your image Tensor :)
                resized_image = np.asarray(image)
               # resized_image = resized_image.transpose(2,0,1).reshape(3,-1)
                #resized_image = resized_image.transpose(1,0)
                #resized_image = resized_image.squeeze()
               # resized_image = resized_image.flatten()
                resized_image = resized_image / 255.

                image_list.append(resized_image)
                labal_hot.append(labal)

            coord.request_stop()
            coord.join(threads)

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
    save_to_file(train_image, train_labal, validation_image, validation_labal, test_image, test_labal)

    return train_image, train_labal, validation_image, validation_labal, test_image, test_labal

# Load image from binary file.
def load_from_file():

    if os.path.exists('FinalDataset/bin/traikn_image.npy') is True:
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
8

# Maim code start here.
train_image,train_labal,validation_image,validation_labal,test_image,test_labal = load_from_file()

# We start
#n_input = 1600
n_input2 = 40
n_output = 50

#x = tf.placeholder(tf.float32, [None, n_input])
x2 = tf.placeholder(tf.float32, [None, 40,40,3 ])
y = tf.placeholder(tf.float32, [None, n_output])

x_tensor = tf.reshape(x2, [-1, 40, 40, 3])
x_tensor2 = x2

# copy code from moudle, and check how good it is.
# Weight matrix is [height x width x input_channels x output_channels]
# Bias is [output_channels]
filter_size = 5
n_filters_1 = 16
n_filters_2 = 16

# parameters
W_conv1 = weight_variable([filter_size, filter_size, 3, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
W_conv3 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv3 = bias_variable([n_filters_2])

# layers
h_conv1 = tf.nn.relu(conv2d(x_tensor2, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
h_conv2_flat = tf.reshape(h_pool2, [-1, 10 * 10 * n_filters_2])


# %% Create the one fully-connected layer:
n_fc = 740
W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([n_fc, n_output])
b_fc2 = bias_variable([n_output])
y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y_pred, y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# We'll train in minibatches and report accuracy:
batch_size = 50
n_epochs = 20
l_loss = list()
for epoch_i in range(n_epochs):
    for batch_i in range(0, len(train_image), batch_size):
        #print('epoch {} barch: {}'.format(epoch_i + 1, batch_i))

        # Calc end of next batch
        end = batch_i+batch_size
        if end > len(train_image):
            end = len(train_image)
        batch_xs = train_image[batch_i:end]
        batch_ys = train_labal[batch_i:end]
        sess.run(optimizer, feed_dict={
            x2: batch_xs, y: batch_ys, keep_prob: 0.5})

    loss = sess.run(accuracy, feed_dict={
                       x2: validation_image,
                       y: validation_labal,
                       keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)

    # Shuffle data.
    train_image, train_labal = Shuffle(train_image, train_labal)

print("Accuracy for test set: {}".format(sess.run(accuracy,
               feed_dict={
                   x2: test_image,
                   y: test_labal,
                   keep_prob: 1.0
               })))

