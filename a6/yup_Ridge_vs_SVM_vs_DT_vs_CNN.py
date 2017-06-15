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



def one_layer_convet(image, kernel):
    output = np.zeros_like(image)

    # get sizes of image and filter/kernel
    (iH, iW) = image.shape
    (kH, kW) = kernel.shape

    # this is how much we need to pad the image to be able to
    # also process convolution at the edges!
    pad = int((kW - 1) / 2)

    return (output[pad:iH, pad:iW])

# Fetch data - caution this is 55MB for the first download
mnist = fetch_mldata('MNIST original', data_home='./data')

# split the dataset into train and test and normalize
# the first 60000 examples already are the training set
split = 60000
X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
Y_train = np.array([int(x) for x in mnist.target[:split]])
#print(type(Y_train))
X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
Y_test = mnist.target[split:]
#print(type(Y_test))

# for speed purpose do not train on all examples
n_train_samples = 5000
n_classes = 10
n_feats = [2,4,6,8,12,16] # for the second layer!

# this is very important here - we select random subset!!!
# this is done to ensure that the minibatches will actually see
# different numbers in each training minibatch!
train_idxs = np.random.randint(0, split-1, n_train_samples)
X_train = X_train[train_idxs, ...]
Y_train = Y_train[train_idxs, ...]
#print(np.shape(X_train))

# 10-fold cross validation
k_fold = KFold(n_splits=10)

result = []
for index, nf in enumerate(n_feats):
    fold_result = []
    print('*** Starting Test of feat [', n_feats[index], ']...')

    # SETUP one-layer CONVnet
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Conv(
                n_feats=nf,
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

    # TRAINING
    for train_indices, valid_indices in k_fold.split(np.array(X_train)):
        np.random.shuffle(train_indices)
        #print(train_indices, valid_indices)
        X_tr = X_train[train_indices, ...]
        Y_tr = Y_train[train_indices, ...]
        X_val = X_train[valid_indices, ...]
        Y_val = Y_train[valid_indices, ...]

        # Train neural network
        t0 = time.time()
        nn.fit(X_tr, Y_tr, learning_rate=0.1, max_iter=15, batch_size=30)
        t1 = time.time()

        # Evaluate on test data
        onelayer_result = nn.error(X_val, Y_val)
        fold_result.append(onelayer_result)

        print('Duration: %.1fs' % (t1 - t0))
        print('Valid error rate: %.4f' % onelayer_result)

    # save the result for each n_feat
    result.append(np.mean(np.array(fold_result)))

print(result)
print('Optimum N_feat Value :', n_feats[result.index(max(result))])

'''

# Try two-layer CONVnet
nn = nnet.NeuralNetwork(
    layers=[
        nnet.Conv(
            n_feats=n_feats,
            filter_shape=(5, 5),
            strides=(1, 1),
            weight_scale=0.1,
        ),
        nnet.Activation('relu'),
        nnet.Pool(
            pool_shape=(2,2),
            strides=(2,2),
            mode='max',
        ),
        nnet.Conv(
            n_feats=n_feats,
            filter_shape=(5,5),
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
nn.fit(X_train, Y_train, learning_rate=0.1, max_iter=10, batch_size=30)
twolayer_result = nn.error(X_test, Y_test)



# Try Ridge Classifier
rcv = RidgeClassifierCV().fit(mnist.data[train_idxs, ...]/255.0, mnist.target[train_idxs])
ridge_result = rcv.score(mnist.data[split:,...]/255.0, mnist.target[split:])

# Try linear svc
lsvc_clf = LinearSVC()
param_grid_lsvc = {'C': [0.1, 1, 10, 100, 1000]}
grid_search_lsvc = GridSearchCV(lsvc_clf, param_grid_lsvc)
lsvc = grid_search_lsvc.fit(mnist.data[train_idxs, ...]/255.0, mnist.target[train_idxs])
linear_result = lsvc.score(mnist.data[split:,...]/255.0, mnist.target[split:])

# Try Decision Tree classifier
dtree_clf = DecisionTreeClassifier()
param_grid_dtree = {'max_depth': [2, 3, 4, 5, 6, None]}
grid_search_dtree = GridSearchCV(dtree_clf, param_grid=param_grid_dtree)
dtc = grid_search_dtree.fit(mnist.data[train_idxs, ...]/255.0, mnist.target[train_idxs])
dtc_result = dtc.score(mnist.data[split:,...]/255.0, mnist.target[split:])


'''
