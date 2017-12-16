from models.NaiveBayes import NaiveBayes
from models.DecisionTree import DecisionTree
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np
import os

DIR = os.path.dirname(__file__)
DATA_TRAIN = os.path.join(DIR, "../data/speeches.train.liblinear")
DATA_TEST = os.path.join(DIR, "../data/speeches.test.liblinear")

# Read in the data
x_train, y_train = load_svmlight_file(DATA_TRAIN)
x_test, y_test = load_svmlight_file(DATA_TEST)

y_train = np.where(y_train==1, 1., 0.)
y_test = np.where(y_test==1, 1., 0.)

np.random.seed(0)
indices = list(range(x_train.shape[1]))
selectedFeatureIndices = np.random.choice(indices, 100)
x_train = x_train.tocsc()[:, selectedFeatureIndices].toarray()
x_test = x_test.tocsc()[:, selectedFeatureIndices].toarray()

## Sample Examples
#np.random.seed(55)
#exampleIndices = np.random.choice(list(range(len(y_train))), size=88, replace=False)
# print "Incdice average: %.2f" % np.mean(exampleIndices)
# x_train = x_train[exampleIndices]
# y_train = y_train[exampleIndices]
# print "Average y_train: %.3f" % np.mean(y_train)

print "Train:"
print x_train.shape
print "Test:"
print x_test.shape

featureNames = [str(index) for index in range(x_train.shape[1])]

model = DecisionTree(maxDepth=3)
model.fit(x_train, y_train, featureNames)

predictions = model.predict(x_test, featureNames)

print "Num Positive"
print np.sum(predictions)
print ""
print "Accuracy"
print np.sum(np.where(predictions == y_test, 1. ,0.)) / float(len(y_test))
print precision_recall_fscore_support(y_test, predictions, average="binary")


