# yup_Ridge_vs_SVM_vs_DT_vs_CNN.py

from sklearn.datasets import fetch_mldata
import numpy as np
import nnet
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from filter_optimizer import optimize_filter
import seaborn as sns

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
#TODO 50000
n_train_samples = 50000
n_classes = 10

# Optimize Filter
#TODO n_train_samples
n_feat1, n_feat2 = optimize_filter(n_train_samples, n_classes, X_train, Y_train, split)

# SETUP two-layers CONVnet
nn = nnet.NeuralNetwork(
    layers=[
        nnet.Conv(
            n_feats=n_feat1,
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
            n_feats=n_feat2,
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

# to compare cnn between with and without weight decay
weight_decay = 0.001
n_feat1, n_feat2 = optimize_filter(500, n_classes, X_train, Y_train, split, weight_decay)
nn_with_weight_decay = nnet.NeuralNetwork(
    layers=[
        nnet.Conv(
            n_feats=n_feat1,
            filter_shape=(5, 5),
            strides=(1, 1),
            weight_scale=0.1,
            weight_decay=weight_decay,
        ),
        nnet.Activation('relu'),
        nnet.Pool(
            pool_shape=(2, 2),
            strides=(2, 2),
            mode='max',
        ),
        nnet.Conv(
            n_feats=n_feat2,
            filter_shape=(5, 5),
            strides=(1, 1),
            weight_scale=0.1,
            weight_decay=weight_decay,
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
CONVnet_result = []
CONVnet_weightdecay_result = []

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

    # Try linear svc
    lsvc_clf = LinearSVC()
    param_grid_lsvc = {'C': [0.1, 1, 10, 100, 1000]}
    grid_search_lsvc = GridSearchCV(lsvc_clf, param_grid_lsvc)
    lsvc = grid_search_lsvc.fit(X_tr, Y_tr)
    linear_result = lsvc.score(X_t, Y_t)
    LinearSVC_result.append(linear_result)

    # Try Decision Tree classifier
    dtree_clf = DecisionTreeClassifier()
    param_grid_dtree = {'max_depth': [2, 3, 4, 5, 6, None]}
    grid_search_dtree = GridSearchCV(dtree_clf, param_grid=param_grid_dtree)
    dtc = grid_search_dtree.fit(X_tr, Y_tr)
    dtc_result = dtc.score(X_t, Y_t)
    DecisionTreeClassifier_result.append(dtc_result)

    # Try CNN
    #TODO max_iter
    nn.fit(X_tr_CNN, Y_tr_CNN, learning_rate=0.1, max_iter=50, batch_size=64)
    CNN_result = 1 - nn.error(X_test, Y_test)
    CONVnet_result.append(CNN_result)

    # Try CNN with weight decay
    # TODO max_iter
    nn_with_weight_decay.fit(X_tr_CNN, Y_tr_CNN, learning_rate=0.1, max_iter=50, batch_size=64)
    CNN_with_weight_decay_result = 1 - nn_with_weight_decay.error(X_test, Y_test)
    CONVnet_weightdecay_result.append(CNN_with_weight_decay_result)

    print('--------------------------------------')
    print('Ridge         [',i+1,']', ridge_result)
    print('SVC           [',i+1,']', linear_result)
    print('DecisionTree  [',i+1,']', dtc_result)
    print('2-layer CNN   [',i+1,']', CNN_result)
    print('2-layer CNN with weight decay  [', i+1, ']', CNN_with_weight_decay_result)
    print('--------------------------------------')


print('** Final Result **')
print(' ')
print('RidgeClassifierCV      :',np.array(RidgeClassifierCV_result),'Mean :', np.mean(np.array(RidgeClassifierCV_result))*100,'(%)', 'Std :', np.std(np.array(RidgeClassifierCV_result)))
print('LinearSVC              :',np.array(LinearSVC_result), 'Mean :', np.mean(np.array(LinearSVC_result))*100,'(%)', 'Std :', np.std(np.array(LinearSVC_result)))
print('DecisionTreeClassifier :',np.array(DecisionTreeClassifier_result), 'Mean :', np.mean(np.array(DecisionTreeClassifier_result))*100,'(%)', 'Std :', np.std(np.array(DecisionTreeClassifier_result)))
print('2-layer CNN            :',np.array(CONVnet_result), 'Mean :', np.mean(np.array(CONVnet_result))*100,'(%)', 'Std :', np.std(np.array(CONVnet_result)))
print('2-layer CNN with weight decay            :',np.array(CONVnet_weightdecay_result), 'Mean :', np.mean(np.array(CONVnet_weightdecay_result))*100,'(%)', 'Std :', np.std(np.array(CONVnet_weightdecay_result)))

'''
pd.DataFrame(data=data[1:,1:],    # values
    index=data[1:,0],    # 1st column as index
    columns=data[0,1:])  # 1st row as the column names
'''

plot_data = pd.DataFrame({'Ridge':RidgeClassifierCV_result, 'SVC':LinearSVC_result, 'DTC':DecisionTreeClassifier_result, 'ConvNet':CONVnet_result, 'ConvNet with weight decay':CONVnet_weightdecay_result})

sns.set_style("whitegrid")
ax = sns.boxplot(data=plot_data)

plt.show()
