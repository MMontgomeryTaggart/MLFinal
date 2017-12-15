from Models.DecisionTree import DecisionTree
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle

data = pickle.load(open("./Data/BOWData.pkl", 'rb'))
numInstances = data["features"].shape[0]
validationFraction = .2
numValidationInstances = int(numInstances * validationFraction)
features = np.array(data["features"])
labels = data["labels"]

indices = np.array(list(range(numInstances)))

np.random.seed(19834)
validationIndices = np.random.choice(indices, size=numValidationInstances, replace=False)
trainingIndices = [index for index in indices if index not in validationIndices]
x_train = features[trainingIndices]
x_test = features[validationIndices]
y_train = labels[trainingIndices]
y_test = labels[validationIndices]

featureNames = [str(index) for index in range(x_train.shape[1])]
model = DecisionTree(maxDepth=10)
model.fit(x_train, y_train, featureNames)

predictions = model.predict(x_test, featureNames)

print precision_recall_fscore_support(y_test, predictions, average="binary")

