from sklearn import datasets
from sklearn import tree
import numpy as np
import pandas
import seaborn as sb
from matplotlib import pyplot as plt

# read the data from sklearn
iris = datasets.load_iris()
# what are the names of the features
iris_fnames = iris.feature_names
# let's convert to pandas DataFrame and name
# the columns according to the feature names
iris_data = pandas.DataFrame(iris.data, columns=iris_fnames)
# the DataFrame should also contain the target values
# so, we should add them here in a column named "Y"
# luckily, extending a dataFrame is easy in pandas:
iris_data["Y"] = pandas.DataFrame(iris.target)

# read the data from the internets using pandas
wine_data = pandas.read_csv(("http://archive.ics.uci.edu/ml/""machine-learning-databases/wine/wine.data"), header=None)
# rename column names to "V1" .. "V14"
wine_data.columns = ["V" + str(i) for i in range(1, len(wine_data.columns) + 1)]
# map the numbers to some nicer label
# label_mapper = {1: 'Label1', 2: 'Label2', 3: 'Label3'}
# and convert array
# wine_data.V1 = wine_data.V1.replace(label_mapper)

# reindex wine's 'V1' position to match like iris_data'Y position
cols = wine_data.columns.tolist()
cols.append(cols.pop(cols.index('V1')))
wine_data = wine_data.reindex(columns=cols)

'''
It takes as input the percentage of each of the 3 IRIS classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays
'''
def shuffleIrisIndices(percentage):
    # all the number of class is same
    class_num = len(iris_data.loc[iris_data.Y == 0])

    # The number of training
    num_shuffle = int(class_num * (percentage / 100))

    # initialize index for each class
    rand_indices_cls0 = np.arange(0, class_num)
    rand_indices_cls1 = rand_indices_cls0 + class_num
    rand_indices_cls2 = rand_indices_cls1 + class_num

    # random shuffle
    np.random.shuffle(rand_indices_cls0)
    np.random.shuffle(rand_indices_cls1)
    np.random.shuffle(rand_indices_cls2)

    # merge each class index for training and testing
    train_indices = np.concatenate((rand_indices_cls0[:num_shuffle], rand_indices_cls1[:num_shuffle],rand_indices_cls2[:num_shuffle]))
    test_indices = np.concatenate((rand_indices_cls0[num_shuffle:], rand_indices_cls1[num_shuffle:], rand_indices_cls2[num_shuffle:]))

    # debug
    # print('[shuffleIrisIndices]')
    # print('~', rand_indices_cls0.max(), num_shuffle, rand_indices_cls0[:num_shuffle])
    # print('~', rand_indices_cls1.max(), num_shuffle, rand_indices_cls1[:num_shuffle])
    # print('~', rand_indices_cls2.max(), num_shuffle, rand_indices_cls2[:num_shuffle])

    return train_indices, test_indices



'''
It takes as input the percentage of each of the 3 wine classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays.
'''
def shuffleWineIndices(percentage):
    # The number of class is different unlike iris data
    label1_num = len(wine_data.loc[wine_data.V1 == 1])
    label2_num = len(wine_data.loc[wine_data.V1 == 2])
    label3_num = len(wine_data.loc[wine_data.V1 == 3])

    # The number of training
    label1_num_shuffle = int(label1_num * (percentage / 100))
    label2_num_shuffle = int(label2_num * (percentage / 100))
    label3_num_shuffle = int(label3_num * (percentage / 100))

    # initialize index for each class
    rand_indices_label1 = np.arange(0, label1_num)
    rand_indices_label2 = np.arange(0, label2_num) + label1_num
    rand_indices_label3 = np.arange(0, label3_num) + label1_num + label2_num

    # random shuffle for each class
    np.random.shuffle(rand_indices_label1)
    np.random.shuffle(rand_indices_label2)
    np.random.shuffle(rand_indices_label3)

    # merge index for training and testing
    train_indices = np.concatenate((rand_indices_label1[:label1_num_shuffle], rand_indices_label2[:label2_num_shuffle], rand_indices_label3[:label3_num_shuffle]))
    test_indices = np.concatenate((rand_indices_label1[label1_num_shuffle:], rand_indices_label2[label2_num_shuffle:], rand_indices_label3[label3_num_shuffle:]))

    # debug
    # print('[shuffleWineIndices]')
    # print('~', rand_indices_label1.max(), label1_num_shuffle, rand_indices_label1[:label1_num_shuffle])
    # print('~', rand_indices_label2.max(), label2_num_shuffle, rand_indices_label2[:label2_num_shuffle])
    # print('~', rand_indices_label3.max(), label3_num_shuffle, rand_indices_label3[:label3_num_shuffle])
    return train_indices, test_indices


'''
Runs the full Naive Bayes implementation for one split of the data into training
and testing. The input consists of the actual trainingData and testData, NOT the indices!
Returns the ERRORS that the classifier makes as PERCENTAGE of len(testData).
'''
def runOneSplitNaiveBayes(trainingData, testData):
    # dictionary of dataFrames for the storage of the probability
    bayesData = {}

    # holds the different decisions
    dV = trainingData.columns[-1]
    decisions = trainingData[dV].unique()

    # holds the independent variables
    iV = trainingData.columns[:-1]

    # training
    # loop through all independent variables
    for c in iV:
        # make an entry in the dictionary with columns consisting of the different decisions
        bayesData[c] = pandas.DataFrame(columns=decisions)
        for d in decisions:
            # Assumption: numeric attribute have a normal or Gaussian probability distribution
            mu = trainingData.loc[trainingData[dV] == d][c].mean(axis=0)
            sigma = trainingData.loc[trainingData[dV] == d][c].std(axis=0)
            bayesData[c][d] = [mu, sigma]

    # testing
    error = 0
    likelihood = {}
    # loop through all test data
    for i, row in testData.iterrows():
        for d in decisions:
            # initialize likelihood
            p = 1
            # collect all likelihoods
            for c in iV:
                x = row[c]
                mu = bayesData[c][d][0]
                sigma = bayesData[c][d][1]
                # compute density value
                f_c_d = np.divide(1, (np.sqrt(2*np.pi)*sigma))*np.exp(np.divide(-((x - mu)**2), (2*(sigma**2))))
                # product of per-attribute likelihood
                p *= f_c_d
            # collect likelihood
            likelihood[d] = p
        # print(max(likelihood.keys(), key=(lambda key: likelihood[key])))
        # find max likelihood's index
        predict = max(likelihood.keys(), key=(lambda key: likelihood[key]))
        actual = row[-1]

        # compare and if different, it's error!
        if predict != actual:
            error += 1

    return error


'''
Runs a simple Decision Tree implementation for one split of the data into
training and testing. The input consists of the actual trainingData and testData,
NOT the indices! In addition the parameter max_depth is handed to the function
that is used to initialize the tree.
'''
def runOneSplitDecisionTree(trainingData, testData, max_depth=None):
    # holds label values
    label = trainingData.iloc[:, -1]

    # training
    # construct a decision tree model using the 1D information gain
    dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)

    # fit it to our data
    dt.fit(trainingData.iloc[:,0:-1], label)

    # test
    error = 0
    # loop through all test data
    for i, row in testData.iterrows():
        # print('decision: ', dt.predict(row.iloc[0:-1])[0], 'with', np.max(dt.predict_proba(row.iloc[0:-1])))

        # compare predict and actual value
        predict = dt.predict(row.iloc[0:-1])[0]
        actual = row[-1]
        # if different, increment error
        if predict != actual:
            error += 1
    return error


# def runOneSplitDecisionTree(trainingData, testData, max_depth=None):
#     # holds label value
#     label = trainingData.iloc[:,-1]
#
#     # holds the different decisions
#     dV = trainingData.columns[-1]
#     decisions = trainingData[dV].unique()
#
#     # holds the independent variables
#     iV = trainingData.columns[:-1]
#
#     # loop through all independent variables
#     for c in iV:
#         for d in decisions:
#             trainingData.loc[trainingData[dV] == d][c]
#         # compute entropy
#
#     error = 0
#     for i, row in testData.iterrows():
#         predict =
#         actual = row[-1]
#         if predict != actual:
#             error += 1
#     return error


# '''
# Main part, run a dual for-loop for each the IRIS data and the WINE data that tests
# different percentages of training and test split and uses 20 repetitions
# for each split to estimate a better error.
# '''
def main():
    # initialize percentage ane repeats
    percentages = [i for i in range(5, 95)]
    reps = np.arange(20)

    p_list = []
    iris_errors_nb = []
    wine_errors_nb = []
    iris_errors_dt = []
    wine_errors_dt = []
    # loop through all percentage
    for p in percentages:
        # repeat
        for i in reps:
            iris_training_indices, iris_test_indices = shuffleIrisIndices(p)
            wine_training_indices, wine_test_indices = shuffleWineIndices(p)

            iris_errorsNB = runOneSplitNaiveBayes(iris_data.loc[iris_training_indices, :], iris_data.loc[iris_test_indices, :])
            wine_errorsNB = runOneSplitNaiveBayes(wine_data.loc[wine_training_indices, :], wine_data.loc[wine_test_indices, :])
            iris_errorsDT = runOneSplitDecisionTree(iris_data.loc[iris_training_indices, :], iris_data.loc[iris_test_indices, :])
            wine_errorsDT = runOneSplitDecisionTree(wine_data.loc[wine_training_indices, :], wine_data.loc[wine_test_indices, :])

            p_list.append(p)
            iris_errors_nb.append(iris_errorsNB)
            wine_errors_nb.append(wine_errorsNB)
            iris_errors_dt.append(iris_errorsDT)
            wine_errors_dt.append(wine_errorsDT)

    # make result dictionary for display
    result = {}
    # merge iris, wine, naivebayes, decision tree
    result["percentage"] = p_list + p_list + p_list + p_list
    result["error"] = iris_errors_nb + iris_errors_dt +  wine_errors_nb + wine_errors_dt
    result["kind"] = ['iris nb' for i in range(0,len(iris_errors_nb))] + ['iris dt' for i in range(0, len(iris_errors_dt))] + ['wine nb' for i in range(0, len(wine_errors_nb))] + ['wine dt' for i in range(0, len(wine_errors_dt))]

    # Plot the result as a very nice plot that has the percentages on the x-axis
    # and for each percentage the mean errors and their confidence intervals around it!
    sb.factorplot(x="percentage", y="error", hue="kind", data=pandas.DataFrame(result), capsize=.2, palette="YlGnBu_d", size=10, aspect=1.5)
    plt.show()


if __name__ == '__main__':
    main()
#
#
# # Which method is better and by how much ?
#
# # What are the storage requirements for the tree versus Naive Bayes in one split?
#
# # Try to decrease max_depth so that you get a tree that is comparable in size to
# # Naive Bayes storage requirements and repeat the experiment from above, making another plot
#
# # Which method is better now and by how much ?
#
