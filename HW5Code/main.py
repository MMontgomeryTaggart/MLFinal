from models.SVM import SVM
from models.LogisticRegression import LogisticRegression
from models.NaiveBayes import NaiveBayes
from models.BaggedForest import BaggedForest
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import numpy as np
import os
import pickle

DIR = os.path.dirname(__file__)
DATA_TRAIN = os.path.join(DIR, "../data/speeches.train.liblinear")
DATA_TEST = os.path.join(DIR, "../data/speeches.test.liblinear")
baggedForestModelPath = "./models/SerializedModels/BaggedForest.pkl"
baggedForestPredictionsTrainPath = "./models/SerializedModels/BaggedForestPredictionsTrain.pkl"
baggedForestPredictionsTestPath = "./models/SerializedModels/BaggedForestPredictionsTest.pkl"

print "Reading the training and test data..."
# Read in the data
x_train, y_train = load_svmlight_file(DATA_TRAIN)
x_test, y_test = load_svmlight_file(DATA_TEST)

print "Train:"
print x_train.shape
print "Test:"
print x_test.shape

#Reshape x_train to match x_test
diff = x_test.shape[1] - x_train.shape[1]
rows = x_train.shape[0]
x_train = hstack((x_train, csr_matrix((rows, diff), dtype=np.float32))).tocsr()
bias_train = np.ones((x_train.shape[0], 1))
bias_test = np.ones((x_test.shape[0], 1))
x_train = hstack((x_train, csr_matrix(bias_train))).tocsr()
x_test = hstack((x_test, csr_matrix(bias_test))).tocsr()


# 1. SVM ********************************************************
print "\n\nSVM ********************************************************"

svmC = [10, 1, .1, .01]
rate = [.001, .00001]
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




# 2. Logistic Regression ********************************************************
print "\n\nLogistic Regression ********************************************************"

tradeoff = [10., 100., 1000., 10000.]
rate = [.001, .0001, .000001]
numEpochs = 3
bestParams = {"params": {}, "accuracy": 0}
configCount = 1
numConfigs = len(tradeoff) * len(rate)
for tradeoffConstant in tradeoff:
    for r in rate:
        print "Evaluating configuration %i of %i. (%.2f%%)" % (
        configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
        print "Tradeoff: %.4f      r: %.2f" % (tradeoffConstant, r)
        configCount += 1
        splitCount = 1
        accuracies = []
        for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(x_train, y_train):
            x_train_split = x_train[trainInds]
            y_train_split = y_train[trainInds]
            x_eval_split = x_train[evalInds]
            y_eval_split = y_train[evalInds]

            model = LogisticRegression(sigma=tradeoffConstant, r=r)
            model.fit(x_train_split, y_train_split)
            predictions = model.predict(x_eval_split)
            accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
            accuracies.append(accuracy)
            print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
            splitCount += 1
        averageAccuracy = np.mean(np.array(accuracies))
        if averageAccuracy > bestParams["accuracy"]:
            bestParams["accuracy"] = averageAccuracy
            bestParams["params"]["sigma"] = tradeoffConstant
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


# 3. Naive Bayes ********************************************************
print "\n\nNaive Bayes ********************************************************"

smoothing = [2.5, 1.5, 1., 0.5]
bestParams = {"params": {}, "accuracy": 0}
configCount = 1
numConfigs = len(smoothing)
for smoothingTerm in smoothing:
    print "Evaluating configuration %i of %i. (%.2f%%)" % (
    configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
    print "Smoothing: %.2f" % smoothingTerm
    configCount += 1
    splitCount = 1
    accuracies = []
    for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(x_train, y_train):
        x_train_split = x_train[trainInds]
        y_train_split = y_train[trainInds]
        x_eval_split = x_train[evalInds]
        y_eval_split = y_train[evalInds]

        model = NaiveBayes(smoothing=smoothingTerm)
        model.fit(x_train_split, y_train_split)
        predictions = model.predict(x_eval_split)
        accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
        accuracies.append(accuracy)
        print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
        splitCount += 1
    averageAccuracy = np.mean(np.array(accuracies))
    if averageAccuracy > bestParams["accuracy"]:
        bestParams["accuracy"] = averageAccuracy
        bestParams["params"]["smoothing"] = smoothingTerm

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



# 3. Bagged Forests ********************************************************
print "\n\nBagged Forests ********************************************************"

model = BaggedForest(numTrees=1000, depth=10)
model.fit(x_train, y_train)

trainingPredictions = model.predict(x_train)
trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
print "Training Accuracy: %f" % trainingAccuracy

predictions = model.predict(x_test)

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy

#pickle.dump(model, open(baggedForestModelPath, "wb"))


# 4. SVM over Bagged Forest ********************************************************

# baggedForestModel = pickle.load(open(baggedForestModelPath, "rb"))
# baggedForestPredictionsTrain = model.predict(x_train, unaveraged=True).T
# baggedForestPredictionsTest = model.predict(x_test, unaveraged=True).T
# pickle.dump(baggedForestPredictionsTrain, open(baggedForestPredictionsTrainPath, "wb"))
# pickle.dump(baggedForestPredictionsTest, open(baggedForestPredictionsTestPath, "wb"))

print "\n\nSVM over Bagged Forest ********************************************************"
baggedForestPredictionsTrain = csr_matrix(pickle.load(open(baggedForestPredictionsTrainPath, "rb")))
baggedForestPredictionsTest = csr_matrix(pickle.load(open(baggedForestPredictionsTestPath, "rb")))

svmC = [10, 1, .1, .01]
rate = [.001, .00001]
numEpochs = 3
bestParams = {"params": {}, "accuracy": 0}
configCount = 1
numConfigs = len(svmC) * len(rate)
for C in svmC:
    for r in rate:
        print "Evaluating configuration %i of %i. (%.2f%%)" % (configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
        print "C: %.4f      r: %.2f" % (C, r)
        configCount += 1
        splitCount = 1
        accuracies = []
        for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(baggedForestPredictionsTrain, y_train):
            x_train_split = baggedForestPredictionsTrain[trainInds]
            y_train_split = y_train[trainInds]
            x_eval_split = baggedForestPredictionsTrain[evalInds]
            y_eval_split = y_train[evalInds]

            model = SVM(C=C, r=r)
            model.fit(x_train_split, y_train_split)
            predictions = model.predict(x_eval_split)
            accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
            accuracies.append(accuracy)
            print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
            splitCount += 1
        averageAccuracy = np.mean(np.array(accuracies))
        if averageAccuracy > bestParams["accuracy"]:
            bestParams["accuracy"] = averageAccuracy
            bestParams["params"]["C"] = C
            bestParams["params"]["r"] = r



print "Best params for %s: (Over Bagged Forest)" % type(model)
print bestParams["params"]
print "Best Average Training Accuracy:"
print bestParams["accuracy"]

bestModel = model.set_params(bestParams["params"])
bestModel.fit(baggedForestPredictionsTrain, y_train)

trainingPredictions = bestModel.predict(baggedForestPredictionsTrain)
trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
print "Training Accuracy: %f" % trainingAccuracy

predictions = bestModel.predict(baggedForestPredictionsTest)

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy

#5. Logistic Regression over Bagged Forest ********************************************************

print "\n\nLogistic Regression over Bagged Forest ********************************************************"

tradeoff = [10., 100., 1000., 10000.]
rate = [.001, .0001, .000001]
numEpochs = 3
bestParams = {"params": {}, "accuracy": 0}
configCount = 1
numConfigs = len(tradeoff) * len(rate)
for tradeoffConstant in tradeoff:
    for r in rate:
        print "Evaluating configuration %i of %i. (%.2f%%)" % (
        configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
        print "Tradeoff: %.4f      r: %.2f" % (tradeoffConstant, r)
        configCount += 1
        splitCount = 1
        accuracies = []
        for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(baggedForestPredictionsTrain, y_train):
            x_train_split = baggedForestPredictionsTrain[trainInds]
            y_train_split = y_train[trainInds]
            x_eval_split = baggedForestPredictionsTrain[evalInds]
            y_eval_split = y_train[evalInds]

            model = LogisticRegression(sigma=tradeoffConstant, r=r)
            model.fit(x_train_split, y_train_split)
            predictions = model.predict(x_eval_split)
            accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
            accuracies.append(accuracy)
            print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
            splitCount += 1
        averageAccuracy = np.mean(np.array(accuracies))
        if averageAccuracy > bestParams["accuracy"]:
            bestParams["accuracy"] = averageAccuracy
            bestParams["params"]["sigma"] = tradeoffConstant
            bestParams["params"]["r"] = r

print "Best params for %s:" % type(model)
print bestParams["params"]
print "Best Average Training Accuracy:"
print bestParams["accuracy"]

bestModel = model.set_params(bestParams["params"])
bestModel.fit(baggedForestPredictionsTrain, y_train)

trainingPredictions = bestModel.predict(baggedForestPredictionsTrain)
trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
print "Training Accuracy: %f" % trainingAccuracy

predictions = bestModel.predict(baggedForestPredictionsTest)

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy