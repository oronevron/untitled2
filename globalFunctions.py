import tensorflow as tf
import numpy as np
import cv2
import math
import os.path
import random
from os import listdir
from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def load_all_image(image_type, load_type, size, with_rot, force_load=False, force_load_val=False):
    train_image, train_label = \
        load_image_new('FinalDataset/jpg_new/Train', 'FinalDataset/bin/Train', image_type, load_type, size, with_rot, force_load)

    validation_image, validation_label = \
        load_image_new('FinalDataset/jpg_new/Validation', 'FinalDataset/bin/Validation', image_type, load_type, size, False, force_load_val)

    test_image, test_label = \
        load_image_new('FinalDataset/jpg_new/Test', 'FinalDataset/bin/Test', image_type, load_type, size, False, force_load_val)

    return train_image,train_label,validation_image,validation_label,test_image,test_label

def Image_split():
    dir_list = listdir('FinalDataset/jpg')

    # For every label
    for i in range(0, len(dir_list)):

        image_list = []
        train_image = []
        validation_image = []
        test_image = []

        dir_images_list = listdir('FinalDataset/jpg/' + dir_list[i])

        # For every image
        for j in range(0, len(dir_images_list)):
            # Read image.
            img = cv2.imread('FinalDataset/jpg/' + str(dir_list[i]) + '/' + str(dir_images_list[j]), 1)
            if (img is None):
                continue

            # Resize image.
            resized_image = img #cv2.resize(img, (size, size))
            # resized_image = resized_image.flatten()
            #resized_image = resized_image / 255.

            image_list.append(resized_image)

        # Shuffle data.
        # random.shuffle(image_list)

        # Calc size of each group(train-validation-test)
        random.shuffle(image_list)

        image_number = len(image_list)
        test_image_size = int(math.ceil(image_number * 0.80))
        validation_image_size = int(math.ceil(image_number * 0.87))

        # Split current label image to train-validation-test
        train_image.extend(image_list[:test_image_size])
        validation_image.extend(image_list[test_image_size:validation_image_size])
        test_image.extend(image_list[validation_image_size:])

        for n in range(0, len(train_image)):
            cv2.imwrite("FinalDataset/jpg_new/Train/" + str(dir_list[i]) + "/" + str(n) + ".jpg", train_image[n])

        for n in range(0, len(validation_image)):
            cv2.imwrite("FinalDataset/jpg_new/Validation/" + str(dir_list[i]) + "/" + str(n) + ".jpg", validation_image[n])

        for n in range(0, len(test_image)):
            cv2.imwrite("FinalDataset/jpg_new/Test/" + str(dir_list[i]) + "/" + str(n) + ".jpg", test_image[n])

def load_image_new(jpg_dir, bin_dir, image_type, load_type, size, with_rot, force_load=False):

    # Try load images from bin
    if force_load == False:
        final_image, final_label = load_bin_new(bin_dir)
        if len(final_image) > 0:
            return final_image, final_label

    image_list = []
    label_hot = []

    dir_list = listdir(jpg_dir)
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
        reader = tf.WholeFileReader()
        # For every label
        for i in range(0,len(dir_list)):

            # Init label one hot vector
            label = np.zeros(51)
            label[int(dir_list[i])] = 1

            dir_images_list = listdir(jpg_dir + "/" + dir_list[i])

            if load_type == 'CV':
                # For every image
                for j in range(0,len(dir_images_list)):
                    # Read image.
                    img = cv2.imread(jpg_dir + "/" + str(dir_list[i]) + '/' + str(dir_images_list[j]),image_type)
                    if (img is None):
                        continue

                    # Resize image.
                    resized_image = cv2.resize(img, (size, size))
                    #resized_image = resized_image.flatten()
                    resized_image = resized_image / 255.

                    image_list.append(resized_image)
                    label_hot.append(label)
            else:
                for j in range(0, len(dir_images_list)):
                    dir_images_list[j] = jpg_dir + "/" + str(dir_list[i]) + '/' + dir_images_list[j]

                filename_queue = tf.train.string_input_producer(dir_images_list)  # list of files to read


                key, value = reader.read(filename_queue)

                if image_type == 0: 
                    channels = 1
                else:
                    channels = 3


                my_img = tf.image.decode_jpeg(value, channels=channels)  # use png or jpg decoder based on your files.
                my_img_re = tf.image.resize_images(my_img, [size, size])

                # ROTATIONAL MANIPULATIONS #

                # # Rotate image left by 90 degrees
                # my_img_rot_1 = tf.image.rot90(my_img_re, 1, name=None)
                #
                # Rotate image left by 180 degrees
                my_img_rot_2 = tf.image.rot90(my_img_re, 2, name=None)
                #
                # # Rotate image left by 270 degrees
                # my_img_rot_3 = tf.image.rot90(my_img_re, 3, name=None)
                #
                # # flip image upside down
                # my_img_flip_vertically = tf.image.flip_up_down(my_img_re)
                #
                # flip image left right
                my_img_flip_horizontally = tf.image.flip_left_right(my_img_re)



                # COLOR MANIPULATION #

                # image with random brightness
                my_img_rnd_brightness = tf.image.random_brightness(my_img_re, 1.0, seed=None)

                my_img_rnd_brightness_2 = tf.image.random_brightness(my_img_re, 1.0, seed=None)

                # image with random contrast
                my_img_rnd_contrast = tf.image.random_contrast(my_img_re, 0.1, 4.0, seed=None)
                my_img_rnd_contrast2 = tf.image.random_contrast(my_img_re, 0.1, 4.0, seed=None)

                # # image with random hue
                # my_img_rnd_hue = tf.image.random_hue(my_img_re, 0.5, seed=None)
                #
                # # image with random saturation
                # my_img_rnd_saturation = tf.image.random_saturation(my_img_re, 0.1, 5.0, seed=None)



                # Start populating the filename queue.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                for m in range(0, len(dir_images_list)):  # length of your filename list
                    # print(str(i) + "," + file_list[m])
                    image = my_img_re.eval()  # here is your image Tensor :)

                    resized_image = np.asarray(image)
                    plt.imshow(np.resize(resized_image.flatten(), [40,40]))

                    # resized_image = resized_image.transpose(2,0,1).reshape(3,-1)
                    # resized_image = resized_image.transpose(1,0)
                    # resized_image = resized_image.squeeze()
                    # resized_image = resized_image.flatten()
                    resized_image = resized_image / 255.

                    image_list.append(resized_image)
                    label_hot.append(label)

                    # if we want rotations
                    if with_rot == True:

                        # if m % 5 == 0:

                            # image = my_img_rot_1.eval()
                            # resized_image = np.asarray(image)
                            # resized_image = resized_image / 255.
                            # image_list.append(resized_image)
                            # label_hot.append(label)

                        # elif m % 5 == 1:
                        #
                        #     image = my_img_rot_2.eval()
                        #     resized_image = np.asarray(image)
                        #     resized_image = resized_image / 255.
                        #     image_list.append(resized_image)
                        #     label_hot.append(label)
                        #
                        # elif m % 5 == 2:
                        #
                        #     image = my_img_rot_3.eval()
                        #     resized_image = np.asarray(image)
                        #     resized_image = resized_image / 255.
                        #     image_list.append(resized_image)
                        #     label_hot.append(label)

                        # if m % 2 == 0:

                            # image = my_img_flip_vertically.eval()
                            # resized_image = np.asarray(image)
                            # resized_image = resized_image / 255.
                            # image_list.append(resized_image)
                            # label_hot.append(label)

                        # elif m % 2 == 1:

                        # the first random brightness
                        # image = my_img_rnd_brightness.eval()
                        image = sess.run(my_img_rnd_brightness)
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        image_list.append(resized_image)
                        label_hot.append(label)

                        # the second random brightness
                        image = my_img_rnd_brightness_2.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        image_list.append(resized_image)
                        label_hot.append(label)

                        # random contrast
                        image = my_img_rnd_contrast.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        image_list.append(resized_image)
                        label_hot.append(label)

                        # the second random contrast
                        image = my_img_rnd_contrast2.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        image_list.append(resized_image)
                        label_hot.append(label)

                        # # random hue
                        # image = my_img_rnd_hue.eval()
                        # resized_image = np.asarray(image)
                        # resized_image = resized_image / 255.
                        # image_list.append(resized_image)
                        # label_hot.append(label)

                        # random saturation
                        # image = my_img_rot_2.eval()
                        # resized_image = np.asarray(image)
                        # resized_image = resized_image / 255.
                        # image_list.append(resized_image)
                        # label_hot.append(label)

    coord.request_stop()
    # coord.join(threads)

            # Shuffle data.
            #random.shuffle(image_list)

            # # Calc size of each group(train-validation-test)
            # image_number = len(image_list)
            # test_image_size = int(math.ceil(image_number * 0.73))
            # validation_image_size = int(math.ceil(image_number * 0.82))

            # Split current label image to train-validation-test
            # final_image.extend(image_list[:test_image_size])
            # final_label.extend(label_hot[:test_image_size])

    # list to numpy array.
    final_image = np.array(image_list)
    final_label = np.array(label_hot)

    # Shuffle data.
    final_image, final_label = Shuffle(final_image, final_label)

    # Save image.
    save_to_file_new(final_image,final_label, bin_dir)

    return final_image,final_label


def load_image(image_type, load_type, size, with_rot):
    train_image = []
    train_label = []
    validation_image = []
    validation_label = []
    test_image = []
    test_label = []

    dir_list = listdir('FinalDataset/jpg')

    # For every label
    for i in range(0,len(dir_list)):

        image_list = []
        label_hot = []

        # Init label one hot vector
        label = np.zeros(51)
        label[int(dir_list[i])] = 1

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
                label_hot.append(label)
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


            # ROTATIONAL MANIPULATIONS #

            # # Rotate image left by 90 degrees
            # my_img_rot_1 = tf.image.rot90(my_img_re, 1, name=None)
            #
            # # Rotate image left by 180 degrees
            # my_img_rot_2 = tf.image.rot90(my_img_re, 2, name=None)
            #
            # # Rotate image left by 270 degrees
            # my_img_rot_3 = tf.image.rot90(my_img_re, 3, name=None)
            #
            # # flip image upside down
            # my_img_flip_vertically = tf.image.flip_up_down(my_img_re)
            #
            # flip image left right
            my_img_flip_horizontally = tf.image.flip_left_right(my_img_re)



            # COLOR MANIPULATION #

            # image with random brightness
            my_img_rnd_brightness = tf.image.random_brightness(my_img_re, 1.0, seed=None)

            my_img_rnd_brightness_2 = tf.image.random_brightness(my_img_re, 1.0, seed=None)

            # image with random contrast
            my_img_rnd_contrast = tf.image.random_contrast(my_img_re, 0.1, 5.0, seed=None)

            # # image with random hue
            # my_img_rnd_hue = tf.image.random_hue(my_img_re, 0.5, seed=None)
            #
            # # image with random saturation
            # my_img_rnd_saturation = tf.image.random_saturation(my_img_re, 0.1, 5.0, seed=None)

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
                    label_hot.append(label)

                    # if we want rotations
                    if with_rot == True:

                        # if m % 5 == 0:

                            # image = my_img_rot_1.eval()
                            # resized_image = np.asarray(image)
                            # resized_image = resized_image / 255.
                            # image_list.append(resized_image)
                            # label_hot.append(label)

                        # elif m % 5 == 1:
                        #
                        #     image = my_img_rot_2.eval()
                        #     resized_image = np.asarray(image)
                        #     resized_image = resized_image / 255.
                        #     image_list.append(resized_image)
                        #     label_hot.append(label)
                        #
                        # elif m % 5 == 2:
                        #
                        #     image = my_img_rot_3.eval()
                        #     resized_image = np.asarray(image)
                        #     resized_image = resized_image / 255.
                        #     image_list.append(resized_image)
                        #     label_hot.append(label)

                        # if m % 2 == 0:

                            # image = my_img_flip_vertically.eval()
                            # resized_image = np.asarray(image)
                            # resized_image = resized_image / 255.
                            # image_list.append(resized_image)
                            # label_hot.append(label)

                        # elif m % 2 == 1:

                        # the first random brightness
                        image = my_img_rnd_brightness.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        train_image.append(resized_image)
                        train_label.append(label)

                        # the second random brightness
                        image = my_img_rnd_brightness_2.eval()
                        resized_image = np.asarray(image)
                        resized_image = resized_image / 255.
                        train_image.append(resized_image)
                        train_label.append(label)

                        # # random contrast
                        # image = my_img_rnd_contrast.eval()
                        # resized_image = np.asarray(image)
                        # resized_image = resized_image / 255.
                        # image_list.append(resized_image)
                        # label_hot.append(label)

                        # random hue
                        # image = my_img_rnd_hue.eval()
                        # resized_image = np.asarray(image)
                        # resized_image = resized_image / 255.
                        # image_list.append(resized_image)
                        # label_hot.append(label)

                        # # random saturation
                        # image = my_img_rnd_saturation.eval()
                        # resized_image = np.asarray(image)
                        # resized_image = resized_image / 255.
                        # image_list.append(resized_image)
                        # label_hot.append(label)

                coord.request_stop()
                coord.join(threads)

        # Shuffle data.
        #random.shuffle(image_list)

        # Calc size of each group(train-validation-test)
        image_number = len(image_list)
        test_image_size = int(math.ceil(image_number * 0.8))
        validation_image_size = int(math.ceil(image_number * 0.87))

        # Split current label image to train-validation-test
        train_image.extend(image_list[:test_image_size])
        train_label.extend(label_hot[:test_image_size])
        validation_image.extend(image_list[test_image_size:validation_image_size])
        validation_label.extend(label_hot[test_image_size:validation_image_size])
        test_image.extend(image_list[validation_image_size:])
        test_label.extend(label_hot[validation_image_size:])

    # list to numpy array.
    train_image = np.array(train_image)
    train_label = np.array(train_label)
    validation_image = np.array(validation_image)
    validation_label = np.array(validation_label)
    test_image = np.array(test_image)
    test_label = np.array(test_label)

    # Shuffle data.
    train_image, train_label = Shuffle(train_image, train_label)
    validation_image, validation_label = Shuffle(validation_image, validation_label)
    test_image, test_label = Shuffle(test_image, test_label)

    # Save image.
    save_to_file(train_image,train_label,validation_image,validation_label,test_image,test_label)

    return train_image,train_label,validation_image,validation_label,test_image,test_label

def load_new_test_image():

    test_new_image = []
    test_new_label = []

    dir_images_list = listdir('FinalDataset/test_jpg')

    # For every label
    for i in range(0,len(dir_images_list)):
        dir_images_list[i] = 'FinalDataset/test_jpg/' + dir_images_list[i]

    # Init label one hot vector
    label = np.zeros(51)
    label[3] = 1

    filename_queue = tf.train.string_input_producer(dir_images_list)  # list of files to read

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    channels = 1

    my_img = tf.image.decode_jpeg(value, channels=channels)  # use png or jpg decoder based on your files.
    my_img_re = tf.image.resize_images(my_img, [40, 40])

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for m in range(0, len(dir_images_list)):  # length of your filename list
            image = my_img_re.eval()  # here is your image Tensor :)
            resized_image = np.asarray(image)
            resized_image = resized_image / 255.

            test_new_image.append(resized_image)
            test_new_label.append(label)

        coord.request_stop()
        coord.join(threads)

    # list to numpy array.
    test_new_image = np.array(test_new_image)
    test_new_label = np.array(test_new_label)

    # Save image.
    # np.save('FinalDataset/bin/test_new_image.npy', test_new_image)
    # np.save('FinalDataset/bin/test_new_label.npy', test_new_label)

    return test_new_image,test_new_label

def load_bin_new(bin_dir):
    image = []
    label = []
    if os.path.exists(bin_dir + "/Image.npy") is True:
        image = np.load(bin_dir + "/Image.npy")
        label = np.load(bin_dir + "/Label.npy")

    return image, label

# Load image from binary file.
def load_from_file(image_type, load_type, size, with_rot,force_load=False):

    if force_load == False and os.path.exists('FinalDataset/bin/train_image.npy') is True:
        train_image = np.load('FinalDataset/bin/train_image.npy')
        train_label = np.load('FinalDataset/bin/train_label.npy')
        validation_image = np.load('FinalDataset/bin/validation_image.npy')
        validation_label = np.load('FinalDataset/bin/validation_label.npy')
        test_image = np.load('FinalDataset/bin/test_image.npy')
        test_label = np.load('FinalDataset/bin/test_label.npy')
    else:
        train_image, train_label, validation_image, validation_label, test_image, test_label \
            = load_image(image_type, load_type, size, with_rot)

    # Return data.
    return train_image, train_label, validation_image, validation_label, test_image, test_label
# def load_image():
#     train_image = []
#     train_label = []
#     validation_image = []
#     validation_label = []
#     test_image = []
#     test_label = []
#
#     # For every label
#     for i in range(0,50):
#
#         image_list = []
#         label_hot = []
#
#         # Init lavel one hot vector
#         label = np.zeros(50)
#         label[i] = 1
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
#             label_hot.append(label)
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
#         train_label.extend(label_hot[:test_image_size])
#         validation_image.extend(image_list[test_image_size:validation_image_size])
#         validation_label.extend(label_hot[test_image_size:validation_image_size])
#         test_image.extend(image_list[validation_image_size:])
#         test_label.extend(label_hot[validation_image_size:])
#
#     # list to numpy array.
#     train_image = np.array(train_image)
#     train_label = np.array(train_label)
#     validation_image = np.array(validation_image)
#     validation_label = np.array(validation_label)
#     test_image = np.array(test_image)
#     test_label = np.array(test_label)
#
#     # Shuffle data.
#     train_image, train_label = Shuffle(train_image, train_label)
#     validation_image, validation_label = Shuffle(validation_image, validation_label)
#     test_image, test_label = Shuffle(test_image, test_label)
#
#     # Save image.
#     save_to_file(train_image,train_label,validation_image,validation_label,test_image,test_label)
#
#     return train_image,train_label,validation_image,validation_label,test_image,test_label
#
# # Load image from binary file.
# def load_from_file():
#
#     if os.path.exists('FinalDataset/bin/train_image.npy') is True:
#         train_image = np.load('FinalDataset/bin/train_image.npy')
#         train_label = np.load('FinalDataset/bin/train_label.npy')
#         validation_image = np.load('FinalDataset/bin/validation_image.npy')
#         validation_label = np.load('FinalDataset/bin/validation_label.npy')
#         test_image = np.load('FinalDataset/bin/test_image.npy')
#         test_label = np.load('FinalDataset/bin/test_label.npy')
#     else:
#         train_image, train_label, validation_image, validation_label, test_image, test_label = load_image()
#
#     # Return data.
#     return train_image, train_label, validation_image, validation_label, test_image, test_label

# Save image to binary file.
def save_to_file_new(image, label, bin_dir):
    np.save(bin_dir + "/Image.npy", image)
    np.save(bin_dir + "/Label.npy", label)

def save_to_file(train_image, train_label, validation_image, validation_label, test_image, test_label):
      np.save('FinalDataset/bin/train_image.npy',train_image)
      np.save('FinalDataset/bin/train_label.npy',train_label)
      np.save('FinalDataset/bin/validation_image.npy',validation_image)
      np.save('FinalDataset/bin/validation_label.npy', validation_label)
      np.save('FinalDataset/bin/test_image.npy',test_image)
      np.save('FinalDataset/bin/test_label.npy',test_label)

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

def confusion_matrix1(actual, prediction, file_name):
    cm = confusion_matrix(actual,prediction)
    df_cm = pd.DataFrame(cm, range(50), range(50))
    sn.set(font_scale=0.90)  # for label size
    fig, ax = plt.subplots(figsize=(17, 12))  # Sample figsize in inches
    cm_plot = sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, linewidths=.5, ax=ax)  # font size
    fig = cm_plot.get_figure()
    fig.savefig('cm/' + str(file_name) + '.png')
    plt.clf()

def accuracy_per_epoch_validation(loss_list_validation, file_name):
    plt.plot(loss_list_validation, color='m')
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.savefig('cm/' + str(file_name) + '.png')
    plt.clf()

def accuracy_per_epoch_train_and_validation(loss_list_train, loss_list_validation, file_name):
    plt.plot(loss_list_train)
    plt.plot(loss_list_validation)
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('cm/' + str(file_name) + '.png')
    plt.clf()

def Restore_and_run_test(model_name):
    a = 1