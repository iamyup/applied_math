from sklearn import datasets
import numpy as np
import pandas
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
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

    # all the number of class is same
    class_num = len(iris.loc[iris[0] == 0])

    # The number of training
    num_shuffle = round(class_num * (percentage / 100)+0.5)

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

    return train_indices, test_indices

'''
It takes as input the percentage of each of the 3 wine classes to use as TRAINING.
Returns the training indices and testing indices for the three classes as two numpy arrays.
'''
def shuffleWineIndices(percentage):
    label1_num = len(wine.loc[wine[0] == 0])
    label2_num = len(wine.loc[wine[0] == 1])
    label3_num = len(wine.loc[wine[0] == 2])

    # The number of training
    label1_num_shuffle = round(label1_num * (percentage / 100))
    label2_num_shuffle = round(label2_num * (percentage / 100))
    label3_num_shuffle = round(label3_num * (percentage / 100))

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
                x = row[j + 1]
                mean, stdev = mv_matrix[i,j]
                if stdev == 0 or calculateProbability(x, mean, stdev) == 0:
                    prior_p += 0
                else:
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
def runOneSplitDecisionTree(trainingData, testData, max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    #print('data', trainingData.iloc[:, 1:])
    #print('label', np.array(trainingData.iloc[:, 0]))
    clf.fit(trainingData.iloc[:,1:], trainingData.iloc[:,0])
    pred = clf.predict(testData.iloc[:,1:])
    #print(pred)
    error = 0
    for i in np.arange(len(pred)):
        if pred[i] != np.array(testData.iloc[:, 0])[i]:
            error += 1

    return 100 * error / testData.shape[0]


'''
Main part, run a dual for-loop for each the IRIS data and the WINE data that tests
different percentages of training and test split and uses 20 repetitions
for each split to estimate a better error.
'''
def main():
    percentages = [3*i+5 for i in range(0, 31)]
    reps = 20
    data_dict = {}
    p_list = []
    err_iris_list_NB = []
    err_wine_list_NB = []
    err_iris_list_DT = []
    err_wine_list_DT = []
    err_iris_list_DT_max = []
    err_wine_list_DT_max = []
    for p in percentages:
        #if __name__ == '__main__':
        for i in np.arange(reps):
            trainingData_iris, testData_iris = shuffleIrisIndices(p)
            trainingData_wine, testData_wine = shuffleWineIndices(p)
            #print('>>> Traing Size :',p,'(%)','| REPS :',i)

            err_iris_NB = runOneSplitNaiveBayes(iris.iloc[trainingData_iris], iris.iloc[testData_iris])
            err_wine_NB = runOneSplitNaiveBayes(wine.iloc[trainingData_wine], wine.iloc[testData_wine])
            err_iris_list_NB.append(err_iris_NB)
            err_wine_list_NB.append(err_wine_NB)
            #print('[NB] IRIS Error : %.2f' % round(err_iris_NB, 2), '(%)',
            #      '|  WINE Error : %.2f' % round(err_wine_NB, 2), '(%)')

            err_iris_DT = runOneSplitDecisionTree(iris.iloc[trainingData_iris], iris.iloc[testData_iris], max_depth=None)
            err_wine_DT = runOneSplitDecisionTree(wine.iloc[trainingData_wine], wine.iloc[testData_wine], max_depth=None)
            err_iris_list_DT.append(err_iris_DT)
            err_wine_list_DT.append(err_wine_DT)
            #print('[DT] IRIS Error : %.2f' % round(err_iris_DT, 2), '(%)',
            #      '|  WINE Error : %.2f' % round(err_wine_DT, 2), '(%)')

            err_iris_DT = runOneSplitDecisionTree(iris.iloc[trainingData_iris], iris.iloc[testData_iris], max_depth=3)
            err_wine_DT = runOneSplitDecisionTree(wine.iloc[trainingData_wine], wine.iloc[testData_wine], max_depth=3)
            #print('[DT] IRIS Error : %.2f' % round(err_iris_DT, 2), '(%)', '|  WINE Error : %.2f' % round(err_wine_DT, 2), '(%)')
            #print(' ')
            err_iris_list_DT_max.append(err_iris_DT)
            err_wine_list_DT_max.append(err_wine_DT)

            p_list.append(p)
        print('>>> Traing Size :', p, '(%)')
        print('[NB] IRIS Error : %.2f' % round(err_iris_list_NB[-1], 2), '(%)','|  WINE Error : %.2f' % round(err_wine_list_NB[-1], 2), '(%)')
        print('[DT] IRIS Error : %.2f' % round(err_iris_list_DT[-1], 2), '(%)','|  WINE Error : %.2f' % round(err_wine_list_DT[-1], 2), '(%)')
        print('[limited_DT] IRIS Error : %.2f' % round(err_iris_list_DT_max[-1], 2), '(%)','|  WINE Error : %.2f' % round(err_wine_list_DT_max[-1], 2), '(%)')
        print('-'*20)

    # Save into Dictionary for plotting
    data_dict['Train Size(%)'] = p_list
    data_dict['IRIS NB Error(%)'] = err_iris_list_NB
    data_dict['WINE NB Error(%)'] = err_wine_list_NB
    data_dict['IRIS DT Error(%)'] = err_iris_list_DT
    data_dict['WINE DT Error(%)'] = err_wine_list_DT
    data_dict['IRIS DT(max=3) Error(%)'] = err_iris_list_DT_max
    data_dict['WINE DT(max=3) Error(%)'] = err_wine_list_DT_max

    # Plot
    sb.set_style("whitegrid", {"xtick.major.size": "10"})
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}
    sb.set_context("paper", rc=paper_rc)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Naive Bayes VS Decision Tree", fontsize=24)

    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)

    ax1.set_title("Naive Bayes for IRIS", fontsize=16)
    ax2.set_title("Naive Bayes for WINE", fontsize=16)
    ax3.set_title("Decision Tree for IRIS", fontsize=16)
    ax4.set_title("Decision Tree for WINE", fontsize=16)
    ax5.set_title("Limited DT for IRIS", fontsize=16)
    ax6.set_title("Limited DT for WINE", fontsize=16)

    sb.factorplot(x="Train Size(%)", y="IRIS NB Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="hls", size=8, aspect=1.2, ax=ax1)
    plt.close()
    sb.factorplot(x="Train Size(%)", y="WINE NB Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="hls", size=8, aspect=1.2, ax=ax2)
    plt.close()
    sb.factorplot(x="Train Size(%)", y="IRIS DT Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="deep", size=8, aspect=1.2, ax=ax3)
    plt.close()
    sb.factorplot(x="Train Size(%)", y="WINE DT Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="deep", size=8, aspect=1.2, ax=ax4)
    plt.close()
    sb.factorplot(x="Train Size(%)", y="IRIS DT(max=3) Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="Oranges", size=8, aspect=1.2, ax=ax5)
    plt.close()
    sb.factorplot(x="Train Size(%)", y="WINE DT(max=3) Error(%)", data=pandas.DataFrame(data_dict),
                       capsize=.2, palette="Oranges", size=8, aspect=1.2, ax=ax6)
    plt.close()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

# Which method is better and by how much ?
'''
For the proper training size of 80%, NB(2.5% error) is about five times better than DT(12.5%).
and moreover, the deviation of error of NB is even smaller than DT.
'''

# What are the storage requirements for the tree versus Naive Bayes in one split?
'''
We think, in case of 'one split', the error rate is needed.
'''

# Try to decrease max_depth so that you get a tree that is comparable in size to Naive Bayes storage requirements
# and repeat the experiment from above, making another plot
'''
when we limit depth as 3, we could see the performance got worse(of course).
and they show larger deviation via repetitions.
we plotted all in one figure to compare at one sight.
and we set p_step as 3% from 5 % to 95 % to save your time!? but it still takes long. :)

yes, The Naive Bayes classifier relies on a small number of parameters to make predictions.
(https://www.quora.com/Natural-Language-Processing-Why-does-Naive-Bayes-have-such-low-storage-requirements)

Thank you very much.
'''
