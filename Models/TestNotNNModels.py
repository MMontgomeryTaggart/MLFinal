from NotNN.BaggedForest import BaggedForest
from NotNN.SVM import SVM
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle

PICKLE_DATA_PATH = "../Data/BOWData.pkl"
TEST_PICKLE_DATA_PATH = "../Data/TestBOWData.pkl"

#Unpickle the data
data = pickle.load(open(PICKLE_DATA_PATH, 'rb'))
testData = pickle.load(open(TEST_PICKLE_DATA_PATH, 'rb'))

x_train = data["features"]
y_train = data["labels"]
x_test = testData["features"]
y_test = testData["labels"]

# Test SVM
svmC = [1000, 100, 1, .1, .01]
rate = [.1, .01, .001, .00001]
numEpochs = 15
bestParams = {"params": {}, "accuracy" : 0}
configCount = 1
numConfigs = len(svmC) * len(rate)
for C in svmC:
    for r in rate:
        print "Evaluating configuration %i of %i. (%.2f%%)" % (configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
        #print "C: %.4f    r: %.4f" % (C, r)
        print "C: %.4f      r: %.2f" % (C, r)
        configCount += 1
        splitCount = 1
        accuracies = []
        for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(x_train, y_train):
            x_train_split = x_train[trainInds]
            y_train_split = y_train[trainInds]
            x_eval_split = x_train[evalInds]
            y_eval_split = y_train[evalInds]

            model = SVM(C=C, r=r)
            model.fit(x_train_split, y_train_split, epochs=numEpochs)
            predictions = model.predict(x_eval_split)
            accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
            accuracies.append(accuracy)
            print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)

            splitCount += 1
        print ""
        averageAccuracy = np.mean(np.array(accuracies))
        if averageAccuracy > bestParams["accuracy"]:
            bestParams["accuracy"] = averageAccuracy
            bestParams["params"]["C"] = C
            bestParams["params"]["r"] = r



print "Best params for %s:" % type(model)
print bestParams["params"]
print "Best Average Training Accuracy:"
print bestParams["accuracy"]

bestModel = model.set_params(bestParams["params"])
bestModel.fit(x_train, y_train)

trainingPredictions = bestModel.predict(x_train)
trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
print "Training Accuracy: %f" % trainingAccuracy

predictions = bestModel.predict(x_test)

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy