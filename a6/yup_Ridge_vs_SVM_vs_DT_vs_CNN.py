from sklearn.datasets import fetch_mldata
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import sys
sys.path.append("/Users/yup/anaconda/lib/python3.6/site-packages/nnet-0.1-py3.6-macosx-10.7-x86_64.egg")
import nnet


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




