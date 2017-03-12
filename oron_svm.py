from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# %matplotlib inline
import globalFunctions

# read data
# mnist = fetch_mldata("MNIST original")

eta = 0.1
# x_train, y_train = mnist.data[:60000] / 255., mnist.target[:60000]
# x_train_shuffle, y_train_shuffle = shuffle(x_train, y_train, random_state=1)
# x_train_shuffle_cut = x_train_shuffle[:10000]
# y_train_shuffle_cut = y_train_shuffle[:10000]
#
# x_test, y_test = mnist.data[60001:] / 255., mnist.target[60001:]
# x_test_shuffle, y_test_shuffle = shuffle(x_test, y_test, random_state=1)
# x_test_shuffle_cut = x_test_shuffle[:1000]
# y_test_shuffle_cut = y_test_shuffle[:1000]

train_image,train_label,validation_image,validation_label,test_image,test_label = globalFunctions.load_from_file(0, 'CV', 40, True, force_load=False)
nsamples, nx, ny = train_image.shape
train_image = train_image.reshape((nsamples,nx*ny))
nsamples, nx, ny = test_image.shape
test_image = test_image.reshape((nsamples,nx*ny))

train_label = globalFunctions.one_hot_to_label_array(train_label)
test_label = globalFunctions.one_hot_to_label_array(test_label)




train_size = 10000
train_image = train_image[:train_size]
train_label = train_label[:train_size]
# test_size = train_size / 5
# test_image = test_image[:test_size]
# test_label = test_label[:test_size]



# Accuracy for test set: 0.361256544503
clf = svm.SVC(gamma=0.001, C=100, kernel='rbf', verbose=False)
# clf = svm.SVC(gamma=0.001, C=3, kernel='poly', degree=5, verbose=False)
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

# clf.fit(x_train_shuffle_cut, y_train_shuffle_cut)
clf.fit(train_image, train_label)

# predict_results =  clf.predict(x_test_shuffle_cut)
predict_results =  clf.predict(test_image)

# predict
# print 'Y HAT: ', clf.predict(x_test_shuffle_cut)
# print "_______________________________________________________________________________________________________________________"
# print 'Y: ', y_test_shuffle_cut
# print "_______________________________________________________________________________________________________________________"

# cm = confusion_matrix(y_test_shuffle_cut, predict_results);
cm = confusion_matrix(test_label, predict_results);

# the y-axis is the target label
# the x-axis is the prediction label
print(cm)

globalFunctions.confusion_matrix1(test_label, predict_results, "svm")
print "_______________________________________________________________________________________________________________________"
# accuracy = accuracy_score(y_test_shuffle_cut, predict_results)
accuracy = accuracy_score(test_label, predict_results)
print ("Accuracy for test set: " + str(accuracy))










# b = y_train[9900:10000]
# c = x_train[9900:10000]

# # show some images
# plt.figure(1);
# for i in range (1,26):
#     ax = plt.subplot(5,5,i);
#     ax.axis('off');
#     if b[i] > 0:
#         ax.imshow(c[i].reshape(28,28), cmap="gray");
#     else:
#         ax.imshow(255-c[i].reshape(28,28), cmap="gray");
# plt.show();



# define svm parameters









# x = [ex for ex, ey in zip(X, Y) if ey == 1 or ey == 8]
#
# print y_test[y_test > 0]
#
# # convert 1 to +1 and 8 to -1
# y = [1 if ey == 1 else -1 for ex, ey in zip(X, Y) if ey == 1 or ey == 8]
# # suffle examples
# x, y = shuffle(x, y, random_state=1)