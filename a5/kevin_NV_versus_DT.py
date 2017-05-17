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
labelMapper = {1:0, 2:1, 3:2} # map the numbers to some nicer label
wine[0]=wine[0].replace(labelMapper) # and convert array

# The likelihood of the features is assumed to be Gaussian.
def calculateProbability(x, mean, stdev):
	exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
	return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent

'''
It takes as input the percentage of each of the 3 IRIS classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays
'''
def shuffleIrisIndices(percentage):
    len_indices = iris.shape[0]
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
    len_indices = len(wine)
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
    #print(byClass.describe())
    m, n = np.shape(byClass.mean())
    mv_matrix = np.zeros((m, n, 2))

    for i in np.arange(m):
        for j in np.arange(n):
            mean = byClass.mean().iloc[i, j]
            stdev = byClass.std().iloc[i, j]
            mv_matrix[i, j] = np.array([mean, stdev])

    #print(mv_matrix)
    #print(testData)
    #print(trainingData)
    error = 0
    for index, row in testData.iterrows():
        results = []
        for i in np.arange(m):
            prior_p = 1 / m
            for j in np.arange(len(row)-1):
                mean, stdev = mv_matrix[i,j]
                x = row[j+1]
                #print(x,mean,stdev)
                prior_p += np.log(calculateProbability(x, mean, stdev))
                #print('class:',i,'--',np.log(calculateProbability(x, mean, stdev)), prior_p)
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
    err_iris_list = []
    err_wine_list = []
    for p in percentages:
        #if __name__ == '__main__':
        for i in np.arange(reps):
            trainingData_iris, testData_iris = shuffleIrisIndices(p)
            trainingData_wine, testData_wine = shuffleWineIndices(p)
            err_iris = runOneSplitNaiveBayes(iris.iloc[trainingData_iris], iris.iloc[testData_iris])
            err_wine = runOneSplitNaiveBayes(wine.iloc[trainingData_wine], wine.iloc[testData_wine])

            print('|Traing Set :',p,'(%)','|  IRIS Error Rate : %.2f' % round(err_iris,2),'(%)','|  WINE Error Rate : %.2f' % round(err_wine,2),'(%)')
            #print('|Traing Set Percentage :', p, '| WINE Error Rate :', err_wine)
            #errorsNB = runOneSplitNaiveBayes(trainingData, testData)
            #errorsDT = runOneSplitDecisionTree(trainingData, testData)
            p_list.append(p)
            err_iris_list.append(err_iris)
            err_wine_list.append(err_wine)
    data_dict['Train Set(%)'] = p_list
    data_dict['IRIS Error(%)'] = err_iris_list
    data_dict['WINE Error(%)'] = err_wine_list
    #print(len(p_list),len(err_iris_list),len(err_wine_list))
    #sb.lmplot(x="Train %", y="Error", data=pandas.DataFrame(data_dict))
    sb.set_style("whitegrid")
    sb.factorplot(x="Train Set(%)", y="IRIS Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="Blues", size=10, aspect=1.5)
    sb.factorplot(x="Train Set(%)", y="WINE Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="Oranges", size=10, aspect=1.5)
    plt.show()


if __name__ == '__main__':
    main()

# Which method is better and by how much ?

# What are the storage requirements for the tree versus Naive Bayes in one split?

# Try to decrease max_depth so that you get a tree that is comparable in size to
# Naive Bayes storage requirements and repeat the experiment from above, making another plot

# Which method is better now and by how much ?
