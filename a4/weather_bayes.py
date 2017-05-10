# Assignment4 - Second task
'''
Second, the naive Bayes implementation of the weather data.
Load the data, copy the stats part where I fill bayesData with the likelihoods into the script
and implement the testing function testDay exactly as shown in the fragment.
Make sure that it returns the tuple exactly as shown in class!
Next, implement a simple for-loop
(hint: you need a special part of pandas to easily iterate over rows in the data!) that returns the results for all rows of "data".
How many errors do you make? So, is naive Bayes better than 1R??
Insert these observations as comments into your script.
'''
import numpy as np
import pandas
import matplotlib.pyplot as plt


# read in our data - taking care to check that it is tab-delimited!
#data = pandas.read_csv('C:/Users/kevin/PycharmProjects/applied math/data/dataWeather.txt', delimiter='\t')
data = pandas.read_csv('data/dataWeather.txt', delimiter='\t')

# Naive Bayes implementation
# let's make a dictionary of dataFrames for the storage
# of the probabilities
bayesData={}

# this holds the different decisions we have
# I'm assuming they live in the last column of the data
dV = data.columns[-1]
decisions = data[dV].unique()

# how many occurrences of each decision do we have?
# this one is used to measure P(yes) and P(no)
decisionsCount = data[dV].value_counts()

# this holds the independent variables (attributes)
iV=data.columns[:-1]

# now loop through all independent variables (attributes)
for c in iV:
    # get the different values we have for each attribute
    values = data[c].unique()
    # make an entry in the dictionary with columns consisting
    # of the different decisions, indexed by the values of
    # the attribute
    bayesData[c]=pandas.DataFrame(columns=decisions,index=values)

    # now loop through all values of the attribute
    for v in values:
        # find out the decisions for that value
        tmp = data[dV][data[c] == v]
        # loop through all decision values
        for d in decisions:
            # determine the likelihood of that combination
            bayesData[c][d][v] = len(tmp[tmp == d])/decisionsCount[d]

# print out the likelihoods
# for c in iV:
#     print(c,'\n',bayesData[c],'\n\n')

# test with a new day
newDay = ['Sunny', 'Cool', 'High', True]

# # test all decision categories
# for d in decisions:
#     # initialize likelihood
#     p=1
#     # collect all likelihoods from table
#     for n,c in enumerate(iV):
#         p=p*bayesData[c][d][newDay[n]]
#     # multiply by the prior, i.e. likelihood of the decision itself
#     p=p*decisionsCount[d]/decisionsCount.sum()
#     print(d,':',p)


# function definition
def testDay(newDay, verbose=False):
    # initialize probabilities to a numpy array of ones of the correct size
    p = np.ones(len(decisions))

    # loop through all possible decisions and get index as well
    for nd, d in enumerate(decisions):
        p[nd] = 1
        # collect all probabilities
        for n, c in enumerate(iV):
            p[nd] = p[nd] * bayesData[c][d][newDay[n]]
        # multiply by the likelihood of the decision itself
        p[nd] = p[nd] * decisionsCount[d] / decisionsCount.sum()
        if verbose:
            print('{:s}: {:.5f}'.format(d, p[nd]))

    # normalize the array so that probabilities sum up to 1
    p = p / p.sum()

    #return the highest probability and the decision to the user
    return max(p), decisions[np.argmax(p)]


for index, day in data.iterrows():
    print(testDay(day[:-1]), day[-1])


# Observations
# How many errors do you make ?
# there is 1 error
# Is naive Bayes better than 1R ?
# Yes. 1R's minimum error is 4. naive Bayes's error is 1.

