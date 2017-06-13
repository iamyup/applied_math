from sklearn.datasets import fetch_mldata
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import nnet
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


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
# the first 60000 exmples already are the training set
split = 60000

X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
Y_train = mnist.target[:split]
X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
Y_test = mnist.target[split:]

# for speed purpose do not train on all examples
n_train_samples = 50000

# this is very important here - we select random subset!!!
# this is done to ensure that the minibatches will actually see
# different numbers in each training minibatch!
train_idxs = np.random.randint(0, split-1, n_train_samples)
X_train = X_train[train_idxs, ...]
Y_train = Y_train[train_idxs, ...]


n_classes = 10
# Try one-layer CONVnet
nn = nnet.NeuralNetwork(
    layers = [
        nnet.Conv(
            n_feats=12,
            filter_shape=(5,5),
            strides=(1,1),
            weight_scale = 0.1,
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

nn.fit(X_train, Y_train, learning_rate=0.1, max_iter=50, batch_size=30)
onelayer_result = nn.error(X_test, Y_test)

n_feats = [2,4,6,8,12,16] # for second layer!
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
        nnet.Activation('relu')
        nnet.Flatten(),
        nnet.Linear(
            n_out=n_classes,
            weight_scale=0.1,
        ),
        nnet.LogRegression(),
    ],
)
nn.fit(X_train, Y_train, learning_rate=0.1, max_iter=50, batch_size=30)
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




