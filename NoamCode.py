import tensorflow as tf
import os
from PIL import Image
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from random import randint



def one_hot(i):
    a = np.zeros(51, 'float32')
    a[i] = 1
    return a


class_folders = os.listdir("/home/user/noam/AllData/")

train_labels = []
train_images = []
test_images = []
test_labels = []
validation_images = []
validation_labels = []

image_size = [28, 28]
batch_size = 50
batch_counter = 0
train_percent = 0.7
test_percent = 0.2
validation_percent = 0.1

for folder_name in class_folders:
    files = os.listdir("/home/user/noam/AllData/" + folder_name)
    folder_images = []#only the images of one folder
    folder_labels = []
    for file in files:

        image = Image.open("/home/user/noam/AllData/" + folder_name + "/" + file).convert('L')  # converting to grayscale
        image = image.resize(image_size, Image.ANTIALIAS)#resizing the picture
        img_array = np.array(image)#turning the picture to list in python
        img_array = np.reshape(img_array,[784])#reshape the list to one dimension
        img_array = img_array / 255#turning the pixels to be between 0 -1

        folder_images.append(img_array)
        folder_labels.append(one_hot(int(folder_name)))


    folder_len = len(folder_images)

    train_size_in_folder=math.floor(train_percent * folder_len)#size of train in one folder
    test_size_in_folder=math.floor(test_percent*folder_len)
    validation_size_in_folder=math.floor(validation_percent*folder_len)

    train_images.extend(folder_images[:train_size_in_folder])
    train_labels.extend(folder_labels[:train_size_in_folder])

    validation_images.extend(folder_images[train_size_in_folder:train_size_in_folder+validation_size_in_folder])
    validation_labels.extend(folder_labels[train_size_in_folder:train_size_in_folder+validation_size_in_folder])

    test_images.extend(folder_images[train_size_in_folder + test_size_in_folder:])
    test_labels.extend(folder_labels[train_size_in_folder + test_size_in_folder:])

final_test_folders =  os.listdir("/home/user/noam/TestData/")
final_test_files = os.listdir("/home/user/noam/TestData/")
final_test_images = []
final_test_im = []
final_test_labels = []


for final_test_file in final_test_files:
    final_test_image = Image.open("/home/user/noam/TestData/" + (final_test_file)).convert('L') # converting to grayscale
    final_test_image = final_test_image.resize(image_size, Image.ANTIALIAS)  # resizing the picture
    final_img_array = np.array(final_test_image)  # turning the picture to list in python
    final_img_array = np.reshape(final_img_array, [784])  # reshape the list to one dimension
    final_img_array = final_img_array / 255  # turning the pixels to be between 0 -1

    final_test_images.append(final_img_array)

    n = (str(final_test_file)).split('.')[0]

    final_test_labels.append(one_hot((int(n)) - 100))

def get_next_batch_train(batch_i):
    global batch_size

    batch_images = train_images[batch_i: batch_i + batch_size]
    batch_labels = train_labels[batch_i: batch_i + batch_size]


    return batch_images, batch_labels




# We will define two layers of convulotionals with ReLU activation function.
#  Every convulotional layer will be followed by 2x2 max-pooling layer.
# Lastly, we will add two fully connected layers (same as in model 2) on top of the convulotional layers and dropout layer between.
# We will optimize the cross-entropy loss function using Adam.

# We start
n_input = 784
n_output = 51

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
# Given tensor, this operation returns a tensor that has the same values as tensor with shape shape.
# If one component of shape is the special value -1, the size of that dimension is computed so that the total size remains constant.
# reshape the tensor to be 28*28*1*some number which will leave the tensor in its shape
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

# functions for convolutions and pooling
def conv2d(x, W):
    # tf.nn.conv2d(input, filter, strides, padding)
    # compute a convolution on input with the filter and given strides(jumps),padding-if the image sizes stays the same
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # performs max pooling on image(reduce the size of the image in 2 in each dimension)
    # ksize the size of the window,strides-how many jumps between windows
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# functions for parameter initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#initialize wight randomly with normal distribution
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import pandas as pd

def print_confusion_matrix(plabels,tlabels):


    plabels = pd.Series(plabels)
    tlabels = pd.Series(tlabels)

    # draw a cross tabulation...
    df_confusion = pd.crosstab(tlabels,plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)

    #print df_confusion
    return df_confusion
def confusionMatrix(text,Labels,y_pred, not_partial):
   # y_actu = np.where(Labels[0]==1)[0]

   dat = []
   for i in Labels:
    a = np.where(i == 1)[0]
    dat.append(a[0])

   y_actu = np.array(dat)
   df = print_confusion_matrix(y_pred,y_actu)


    #print plt.imshow(df.as_matrix())
   if not_partial:
     writer_full_mat = pd.ExcelWriter('FullMat.xlsx')
     df.to_excel(writer_full_mat, 'Sheet1')
     writer_full_mat.save()
     print ("\n",classification_report(y_actu, y_pred))
   else:
       writer_part_mat = pd.ExcelWriter('PartialMat.xlsx')
       df.to_excel(writer_part_mat, 'Sheet1')
       writer_part_mat.save()
       print("\n", df)
       print ("\n\t------------------------------------------------------\n")




# Weight matrix is [height x width x input_channels x output_channels]
# Bias is [output_channels]
filter_size = 5
n_filters_1 = 16
n_filters_2 = 16#number of layers of pictures in each layer of neurons

# parameters
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])
b_conv1 = bias_variable([n_filters_1])
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])

# layers
h_conv1 = tf.nn.relu(conv2d(x_tensor, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 7x7 is the size of the image after the convolutional and pooling layers (28x28 -> 14x14 -> 7x7)
h_conv2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * n_filters_2])

# %% Create the one fully-connected layer:
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
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
#batch size already initialized
n_epochs = 5
l_loss = list()

for epoch_i in range(n_epochs):

    # shuffle the train labels and train images toghther
    train_images, train_labels = shuffle(train_images, train_labels)

    for batch_i in range(0, len(train_images), batch_size):
        batch_xs, batch_ys = get_next_batch_train(batch_i)
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})

    loss = sess.run(accuracy, feed_dict={
                       x: validation_images,
                       y: validation_labels,
                       keep_prob: 1.0 })
    print('Validation accuracy for epoch {} is: {}'.format(epoch_i + 1, loss))
    l_loss.append(loss)



# print("Accuracy for test set: {}".format(sess.run(accuracy,
#                feed_dict={
#                    x: final_test_images,
#                    y: final_test_labels,
#                    keep_prob: 1.0
#                })))

predictions = sess.run([correct_prediction], feed_dict={x: test_images, y: test_labels,  keep_prob: 1.0})
prediction = tf.argmax(y_pred, 1)
labels = prediction.eval(feed_dict={x: test_images, y: test_labels, keep_prob: 1.0}, session=sess)

confusionMatrix("Partial Confusion matrix", test_labels, predictions[0], False)  # Partial confusion Matrix
confusionMatrix("Complete Confusion matrix", test_labels, labels, True)  # complete confusion Matr



plt.title('CNN Acuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(l_loss, color='m')
plt.show()


predictions2 = sess.run([correct_prediction], feed_dict={x: final_test_images, y: final_test_labels,  keep_prob: 1.0})


prediction2 = tf.argmax(y_pred, 1)
labels2 = prediction2.eval(feed_dict={x: final_test_images, y: final_test_labels,  keep_prob: 1.0}, session=sess)

confusionMatrix("Partial Confusion matrix", final_test_labels, predictions2[0], False)  # Partial confusion Matrix
confusionMatrix("Complete Confusion matrix", final_test_labels, labels2, True)


print ('Neural Network predicted', predictions2[0], "for your digit")




#images[0] = final_test_images[0]

#my_classification = sess.run(tf.argmax(y, 1), feed_dict={x: [images[0]]})



#with open('confusion.txt', 'w') as f:
#    f.write(np.array2string(confusion_matrix(test_labels, y_pred), separator=', '))


