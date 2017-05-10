
from sklearn import datasets
import pandas
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris()
print(iris)

# what are the names of the features
fnames = iris.feature_names
print(fnames)

# let's convert to pandas DataFrame and name
# the columns according to the feature names
data = pandas.DataFrame(iris.data, columns=fnames)
print(data)

# the DataFrame should also contain the target values
# so, we should add them here in a column named "Y"
# luckily, extending a dataFrame is easy in pandas:
data["Y"] = pandas.DataFrame(iris.target)

# show the pairplot of all variables
# since seaborn by default plots all NUMERIC columns,
# it would plot the "Y"-column as well, so we need to
# pass the "vars" argument to restrict plotting to the feature columns only
g = sb.pairplot(data, vars=fnames, hue="Y", kind="reg")

# construct PCA method
irisPCA = PCA()

# standardize our data (only the actual numeric data)
pcaData = np.array(data.loc[:, fnames[0]:fnames[-1]])
print(pcaData)
pcaDataM = pcaData.mean(axis=0)
print(pcaDataM)
pcaDataS = pcaData.std(axis=0)
dataStd = (pcaData - pcaDataM) / pcaDataS

# fit PCA to data
irisPCA.fit(dataStd)

# now project data onto these dimensions:
dataTrans = irisPCA.transform(dataStd)

# now convert the result back into DataFrame for the first two dimension
dataToPlot = pandas.DataFrame(dataTrans[:,0:2], columns=["Dim1", "Dim2"])


# add the IRIS flowers
dataToPlot["Y"] = data["Y"]

# and plot the result using seaborn without regression
sb.lmplot(x="Dim1", y="Dim2", data=dataToPlot, hue="Y", fit_reg=False, legend_out=True)

# we want to know how much variance we can explain - this is contained somewhere in irisPCA
var_top2 = irisPCA.explained_variance_ratio_[0]+irisPCA.explained_variance_ratio_[1]
plt.title("Two dimensions explain {:.2f}% of variance".format(var_top2*100))

''' COMMENT 1
we can see that petal length and petal width are highly correlated.
'''

plt.show()


# read the data from the internets using pandas
data = pandas.read_csv(("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"), header=None)
print(data)

# rename column names to “V1” .. “V14”
data.columns=["V"+str(i) for i in range(1,len(data.columns)+1)]
print(data.columns)

# map the numbers to some nicer label
labelMapper = {1:'Label1', 2:'Label2', 3:'Label3'}

# and convert array (I have to reassign it!)
data.V1=data.V1.replace(labelMapper)

# get all the chemical variables for each wine into X:
X = data.loc[:, "V2":]
# and the first column is my label (dependent variable)
y = data.V1
# print out subview
print(data.loc[1:20,"V1":"V14"])

h = sb.pairplot(data, vars=data.columns[1:], hue="V1", kind="reg")

# construct PCA method
winePCA = PCA()

# standardize our data (only the actual numeric data)
pcaData = np.array(data.loc[:, "V2":"V14"])
print(pcaData)
pcaDataM = pcaData.mean(axis=0)
print(pcaDataM)
pcaDataS = pcaData.std(axis=0)
dataStd = (pcaData - pcaDataM) / pcaDataS

# fit PCA to data
winePCA.fit(dataStd)

# now project data onto these dimensions:
dataTrans = winePCA.transform(dataStd)

# now convert the result back into DataFrame for the first two dimension
dataToPlot = pandas.DataFrame(dataTrans[:,0:2], columns=["Dim1", "Dim2"])


# add the wine
dataToPlot["V1"] = data["V1"]

# and plot the result using seaborn without regression
sb.lmplot(x="Dim1", y="Dim2", data=dataToPlot, hue="V1", fit_reg=False, legend_out=True)

# we want to know how much variance we can explain - this is contained somewhere in winePCA
var_top2 = winePCA.explained_variance_ratio_[0]+winePCA.explained_variance_ratio_[1]
plt.title("Two dimensions explain {:.2f}% of variance of WINE_DATA".format(var_top2*100))

plt.show()

''' COMMENT 2
We can see from the plot that wine samples of "Label1" have much lower values of "Dim1" than "Label3".
Therefore, "Dim1" separates "Label1" from "Label3" in this new space.

We can also see that "Label2" have much higher values of "Dim2" than "Label 1 and 3".
Therefore, "Dim2" separates "Label2" from others.

Therefore, the new components "Dim1" and "Dim2" are useful to distinguish wine samples of the three different Label.
'''

