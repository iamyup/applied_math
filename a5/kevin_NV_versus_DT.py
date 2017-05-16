from sklearn import datasets
import numpy as np
import pandas
import seaborn as sb
from matplotlib import pyplot as plt
# import math

# load iris data as panda frame
iris_data = datasets.load_iris()
iris = pandas.DataFrame(iris_data.data)
iris.insert(0, 'class', iris_data.target)
iris.columns = range(len(list(iris)))

# load wine data as panda frame
wine = pandas.read_csv(("http://archive.ics.uci.edu/ml/""machine-learning-databases/wine/wine.data"), header=None)

data = iris.copy()
#data = wine.copy()

# The likelihood of the features is assumed to be Gaussian.
def calculateProbability(x, mean, stdev):
	exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
	return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent

'''
It takes as input the percentage of each of the 3 IRIS classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays
'''
def shuffleIrisIndices(percentage):
    len_indices = data.shape[0]
    #print('PRINT LEN :',len_indices)
    #print(int(len_indices * percentage / 100))
    train_indices = np.random.choice(len_indices, int(len_indices * percentage / 100), replace=False)
    test_indices = np.setdiff1d(np.arange(len_indices), train_indices)
    #print(len(train_indices))
    #print(len(test_indices))
    return train_indices, test_indices


'''
It takes as input the percentage of each of the 3 wine classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays.
'''
def shuffleWineIndices(percentage):
    len_indices = len(data)
    #print(len_indices)
    #print(int(len_indices * percentage / 100))
    train_indices = np.random.choice(len_indices, int(len_indices * percentage / 100), replace=False)
    test_indices = np.setdiff1d(np.arange(len_indices), train_indices)
    #print(len(train_indices))
    #print(len(test_indices))
    return train_indices, test_indices



'''
Runs the full Naive Bayes implementation for one split of the data into training
and testing. The input consists of the actual trainingData and testData, NOT the indices!
Returns the ERRORS that the classifier makes as PERCENTAGE of len(testData).
'''

def runOneSplitNaiveBayes(trainingData, testData):
    byClass = trainingData.groupby(0)
    m, n = np.shape(byClass.mean())
    mv_matrix = np.zeros((m, n, 2))

    for i in np.arange(m):
        for j in np.arange(n):
            mean = byClass.mean().iloc[i, j]
            stdev = byClass.std().iloc[i, j]
            mv_matrix[i, j] = np.array([mean, stdev])

    #print(mv_matrix)
    #print(testData)
    error = 0
    for index, row in testData.iterrows():
        results = []
        for i in np.arange(m):
            prior_p = 1 / m
            for j in np.arange(len(row)-1):
                mean, stdev = mv_matrix[i,j]
                x = row[j+1]
                #print(x,mean,stdev)
                prior_p *= calculateProbability(x, mean, stdev)
            results.append(prior_p)
        #print('God, thanks', index, results)
        #print(np.argmax(results),row[0])
        if np.argmax(results) != int(row[0]):
            error += 1
    #print(100 * error / testData.shape[0])
    return 100 * error / testData.shape[0]


'''
Runs a simple Decision Tree implementation for one split of the data into
training and testing. The input consists of the actual trainingData and testData,
NOT the indices! In addition the parameter max_depth is handed to the function
that is used to initialize the tree.
'''
def runOneSplitDecisionTree(trainingData, testData, max_depth=None):
    #print(testData)
    error = 0
    return error


'''
Main part, run a dual for-loop for each the IRIS data and the WINE data that tests
different percentages of training and test split and uses 20 repetitions
for each split to estimate a better error.
'''
def main():

    percentages = [i for i in range(5, 96)]
    reps = 20
    data_dict = {}
    p_list = []
    err_list = []
    for p in percentages:
        #if __name__ == '__main__':
        for i in np.arange(reps):
            trainingData, testData = shuffleIrisIndices(p)
            #trainingData, testData = shuffleWineIndices(p)
            err = runOneSplitNaiveBayes(data.iloc[trainingData], data.iloc[testData])
            #runOneSplitNaiveBayes(wine[trainingData], wine[testData])

            print('--Traing Set Percentage :',p,'--Error Rate :',err)
            #errorsNB = runOneSplitNaiveBayes(trainingData, testData)
            #errorsDT = runOneSplitDecisionTree(trainingData, testData)
            p_list.append(p)
            err_list.append(err)
    data_dict['Train %'] = p_list
    data_dict['Error'] = err_list


    #sb.lmplot(x="Train %", y="Error", data=pandas.DataFrame(data_dict))
    sb.factorplot(x="Train %", y="Error", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="YlGnBu_d", size=10, aspect=1.5)
    plt.show()

    #g.set(xlim=(0, 80), ylim=(-.05, 1.05))


if __name__ == '__main__':
    main()

# Which method is better and by how much ?

# What are the storage requirements for the tree versus Naive Bayes in one split?

# Try to decrease max_depth so that you get a tree that is comparable in size to
# Naive Bayes storage requirements and repeat the experiment from above, making another plot

# Which method is better now and by how much ?