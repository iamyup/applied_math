from sklearn import datasets
import numpy as np
import pandas

iris = datasets.load_iris()
wine = pandas.read_csv(("http://archive.ics.uci.edu/ml/""machine-learning-databases/wine/wine.data"), header=None)

'''
It takes as input the percentage of each of the 3 IRIS classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays
'''
def shuffleIrisIndices(percentage):
    train_indices = 0
    test_indices = 0
    return train_indices, test_indices


'''
It takes as input the percentage of each of the 3 wine classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays.
'''
def shuffleWineIndices(percentage):
    train_indices = 0
    test_indices = 0
    return train_indices, test_indices


'''
Runs the full Naive Bayes implementation for one split of the data into training
and testing. The input consists of the actual trainingData and testData, NOT the indices!
Returns the ERRORS that the classifier makes as PERCENTAGE of len(testData).
'''
def runOneSplitNaiveBayes(trainingData, testData):
    error = 0
    return error


'''
Runs a simple Decision Tree implementation for one split of the data into
training and testing. The input consists of the actual trainingData and testData,
NOT the indices! In addition the parameter max_depth is handed to the function
that is used to initialize the tree.
'''
def runOneSplitDecisionTree(trainingData, testData, max_depth=None):
    error = 0
    return error


'''
Main part, run a dual for-loop for each the IRIS data and the WINE data that tests
different percentages of training and test split and uses 20 repetitions
for each split to estimate a better error.
'''
def main():

    percentages = [i for i in range(5, 95)]
    reps = 20
    for p in percentages:
        if __name__ == '__main__':
            for i in reps:
                errorsNB = runOneSplitNaiveBayes(trainingData, testData)
                errorsDT = runOneSplitDecisionTree(trainingData, testData)

        # Plot the result as a very nice plot that has the percentages on the x-axis
        # and for each percentage the mean errors and their confidence intervals around it!


if __name__ == '__main__':
    main()


# Which method is better and by how much ?

# What are the storage requirements for the tree versus Naive Bayes in one split?

# Try to decrease max_depth so that you get a tree that is comparable in size to
# Naive Bayes storage requirements and repeat the experiment from above, making another plot

# Which method is better now and by how much ?

