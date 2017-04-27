from sklearn import datasets
import pandas
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

iris = datasets.load_iris()

# what are the names of the features
fnames = iris.feature_names

# let's convert to pandas DataFrame and name
# the columns according to the feature names
data = pandas.DataFrame(iris.data, columns=fnames)

# the DataFrame should also contain the target values
# so, we should add them here in a column named "Y"
# luckily, extending a dataFrame is easy in pandas:
data["Y"] = pandas.DataFrame(iris.target)

# show the pairplot of all variables
# since seaborn by default plots all NUMERIC columns,
# it would plot the "Y"-column as well, so we need to
# pass the "vars" argument to restrict plotting to the feature columns only
sb.pairplot(data, vars=fnames, hue="Y", kind="reg")

# construct PCA method
irisPCA = PCA()

# standardize our data (only the actual numeric data)
pcaData = np.array(data.loc[:, fnames[0]:fnames[-1]])
pcaDataM = pcaData.mean(axis=0)
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
sb.lmplot(x="Dim1", y="Dim2", data=dataToPlot, hue="Y", fit_reg=False)

# we want to know how much variance we can explain - this is contained somewhere in irisPCA
plt.title("Two dimensions explain {:.2f}% of variance".format(0))

plt.show()




