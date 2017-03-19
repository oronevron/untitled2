from sklearn.metrics import accuracy_score
from sklearn import svm
# %matplotlib inline
import globalFunctions

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

clf = svm.SVC(gamma=0.001, C=100, kernel='rbf', verbose=False)
# clf = svm.SVC(gamma=0.001, C=3, kernel='poly', degree=5, verbose=False)
# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

clf.fit(train_image, train_label)

predict_results =  clf.predict(test_image)

globalFunctions.confusion_matrix1(test_label, predict_results, "svm")

accuracy = accuracy_score(test_label, predict_results)
print ("Accuracy for test set: " + str(accuracy))