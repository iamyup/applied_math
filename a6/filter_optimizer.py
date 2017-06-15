# yup_Ridge_vs_SVM_vs_DT_vs_CNN.py

from sklearn.datasets import fetch_mldata
import numpy as np
import nnet
import time
from sklearn.model_selection import KFold, cross_val_score

# 10-fold cross validation
k_fold = KFold(n_splits=10)

def optimize_filter(n_train_samples,n_classes,X_train,Y_train, split):
    train_idxs = np.random.randint(0, split - 1, n_train_samples)
    n_feats = [2, 4, 6, 8, 12, 16]  # for the second layer!

    X_train = X_train[train_idxs, ...]
    Y_train = Y_train[train_idxs, ...]

    one_layer_result = []
    for index, nf in enumerate(n_feats):
        fold_result = []
        print('*** Starting 1-layer test of feat [', nf, ']...')

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
            # TODO: max_iter 50
            nn.fit(X_tr, Y_tr, learning_rate=0.1, max_iter=5, batch_size=30)
            t1 = time.time()

            # Evaluate on test data
            error = nn.error(X_val, Y_val)
            fold_result.append(error)

            print('Duration: %.1fs' % (t1 - t0))
            print('Valid error rate: %.4f' % error)

        # save the result for each n_feat
        one_layer_result.append(np.mean(np.array(fold_result)))

    best_one_layer = n_feats[np.argmin(one_layer_result)]

    # Try two-layer CONVnet
    two_layer_result = []
    for index, nf in enumerate(n_feats):
        fold_result = []
        print('*** Starting 2-layers-test of feat [', nf, ']...')

        # SETUP two-layers CONVnet
        nn = nnet.NeuralNetwork(
            layers=[
                nnet.Conv(
                    # optimal parameter for the first layer of the two-layer CNN
                    n_feats=best_one_layer,
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
            # TODO: max_iter 50
            nn.fit(X_tr, Y_tr, learning_rate=0.1, max_iter=15, batch_size=30)
            t1 = time.time()

            # Evaluate on test data
            error = nn.error(X_val, Y_val)
            fold_result.append(error)

            print('Duration: %.1fs' % (t1 - t0))
            print('Valid error rate: %.4f' % error)

        # save the result for each n_feat
        two_layer_result.append(np.mean(np.array(fold_result)))

    best_two_layer = n_feats[np.argmin(two_layer_result)]

    print('One-layer result :', one_layer_result)
    print('Two-layer result :', two_layer_result)
    print('Two-Layer Optimum N_feat Value :', best_one_layer, best_two_layer)

    return best_one_layer, best_two_layer
