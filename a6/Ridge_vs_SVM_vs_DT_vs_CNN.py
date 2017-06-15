# yup_Ridge_vs_SVM_vs_DT_vs_CNN.py

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import nnet
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import time

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_val_score


# Fetch data - caution this is 55MB for the first download
mnist = fetch_mldata('MNIST original', data_home='./data')

# split the dataset into train and test and normalize
# the first 60000 examples already are the training set
split = 60000
X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
Y_train = np.array([int(x) for x in mnist.target[:split]])
X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
Y_test = mnist.target[split:]

# for speed purpose do not train on all examples
n_train_samples = 1000
n_classes = 10

# Optimize Filter
n_feat1, n_feat2 = optimize_filter(5000, n_classes, X_train, Y_train, split)

# SETUP two-layers CONVnet
nn = nnet.NeuralNetwork(
    layers=[
        nnet.Conv(
            n_feats= n_feat1,
            filter_shape=(5, 5),
            strides=(1, 1),
            weight_scale=0.1,
        ),
        nnet.Activation('relu'),
        nnet.Pool(
            pool_shape=(2, 2),
            strides=(2, 2),
            mode='max',
        ),
        nnet.Conv(
            n_feats= n_feat2,
            filter_shape=(5, 5),
            strides=(1, 1),
            weight_scale=0.1,
        ),
        nnet.Activation('relu'),
        nnet.Flatten(),
        nnet.Linear(
            n_out=n_classes,
            weight_scale=0.1,
        ),
        nnet.LogRegression(),
    ],
)

RidgeClassifierCV_result = []
LinearSVC_result = []
DecisionTreeClassifier_result = []

for i in range(20):

    # Data Set
    # this is very important here - we select random subset!!!
    # this is done to ensure that the minibatches will actually see
    # different numbers in each training minibatch!

    train_idxs = np.random.randint(0, split-1, n_train_samples)

    # CNN Data input
    X_tr_CNN = X_train[train_idxs, ...]
    Y_tr_CNN = Y_train[train_idxs, ...]

    # Other Data input
    X_tr = mnist.data[train_idxs, ...]/255.0
    Y_tr = mnist.target[train_idxs]
    X_t = mnist.data[split:, ...] / 255.0
    Y_t = mnist.target[split:]

    # Try Ridge Classifier
    rcv = RidgeClassifierCV().fit(X_tr, Y_tr)
    ridge_result = rcv.score(X_t, Y_t)
    RidgeClassifierCV_result.append(ridge_result)
    print('Ridge         [',i+1,']', ridge_result)

    # Try linear svc
    lsvc_clf = LinearSVC()
    param_grid_lsvc = {'C': [0.1, 1, 10, 100, 1000]}
    grid_search_lsvc = GridSearchCV(lsvc_clf, param_grid_lsvc)
    lsvc = grid_search_lsvc.fit(X_tr, Y_tr)
    linear_result = lsvc.score(X_t, Y_t)
    LinearSVC_result.append(linear_result)
    print('SVC           [',i+1,']', linear_result)

    # Try Decision Tree classifier
    dtree_clf = DecisionTreeClassifier()
    param_grid_dtree = {'max_depth': [2, 3, 4, 5, 6, None]}
    grid_search_dtree = GridSearchCV(dtree_clf, param_grid=param_grid_dtree)
    dtc = grid_search_dtree.fit(X_tr, Y_tr)
    dtc_result = dtc.score(X_t, Y_t)
    DecisionTreeClassifier_result.append(dtc_result)
    print('DecisionTree [',i+1,']', dtc_result)

    # Try CNN
    nn.fit(X_tr_CNN, Y_tr_CNN, learning_rate=0.1, max_iter=15, batch_size=64)

print('******************')
print('** Final Result **')
print('******************')
print('RidgeClassifierCV      :',np.array(RidgeClassifierCV_result),'Mean :', np.mean(np.array(RidgeClassifierCV_result))*100,'(%)', 'Std :', np.std(np.array(RidgeClassifierCV_result)))
print('LinearSVC              :',np.array(LinearSVC_result), 'Mean :', np.mean(np.array(LinearSVC_result))*100,'(%)', 'Std :', np.std(np.array(LinearSVC_result)))
print('DecisionTreeClassifier :',np.array(DecisionTreeClassifier_result), 'Mean :', np.mean(np.array(DecisionTreeClassifier_result))*100,'(%)', 'Std :', np.std(np.array(DecisionTreeClassifier_result)))
