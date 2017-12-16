from NotNN.BaggedForest import BaggedForest
from NotNN.SVM import SVM
from NotNN.LogisticRegression import LogisticRegression
from NotNN.NaiveBayes import NaiveBayes
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pickle
from scipy.sparse import csr_matrix

PICKLE_DATA_PATH = "../Data/BOWData.pkl"
TEST_PICKLE_DATA_PATH = "../Data/TestBOWData.pkl"

#Unpickle the data
data = pickle.load(open(PICKLE_DATA_PATH, 'rb'))
testData = pickle.load(open(TEST_PICKLE_DATA_PATH, 'rb'))

x_train = csr_matrix(data["features"])
y_train = data["labels"]
x_test = csr_matrix(testData["features"])
y_test = testData["labels"]

# # Test SVM ******************************
# svmC = [3000, 1000, 100]
# rate = [.5, .1, .01, .001, .00001]
# numEpochs = 40
# bestParams = {"params": {}, "accuracy" : 0}
# configCount = 1
# numConfigs = len(svmC) * len(rate)
# for C in svmC:
#     for r in rate:
#         print "Evaluating configuration %i of %i. (%.2f%%)" % (configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
#         #print "C: %.4f    r: %.4f" % (C, r)
#         print "C: %.4f      r: %.2f" % (C, r)
#         configCount += 1
#         splitCount = 1
#         accuracies = []
#         for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(x_train, y_train):
#             x_train_split = x_train[trainInds]
#             y_train_split = y_train[trainInds]
#             x_eval_split = x_train[evalInds]
#             y_eval_split = y_train[evalInds]
#
#             model = SVM(C=C, r=r)
#             model.fit(x_train_split, y_train_split, epochs=numEpochs)
#             predictions = model.predict(x_eval_split)
#             accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
#             accuracies.append(accuracy)
#             print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
#
#             splitCount += 1
#         print ""
#         averageAccuracy = np.mean(np.array(accuracies))
#         if averageAccuracy > bestParams["accuracy"]:
#             bestParams["accuracy"] = averageAccuracy
#             bestParams["params"]["C"] = C
#             bestParams["params"]["r"] = r
#
#
#
# print "Best params for %s:" % type(model)
# print bestParams["params"]
# print "Best Average Training Accuracy:"
# print bestParams["accuracy"]
#
# bestModel = model.set_params(bestParams["params"])
# bestModel.fit(x_train, y_train)
#
# trainingPredictions = bestModel.predict(x_train)
# trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
# print "Training Accuracy: %f" % trainingAccuracy
#
# predictions = bestModel.predict(x_test)
#
# accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
# print "Test Accuracy: %f" % accuracy

# 2. Logistic Regression ********************************************************
print "\n\nLogistic Regression ********************************************************"

tradeoff = [10., 100., 1000., 10000.]
rate = [.1, .01, .001]
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

rawTrainingPredictions = bestModel.predict(x_train)
trainingPredictions = np.where(rawTrainingPredictions == 1., 1., 0.)
trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
print "Training Accuracy: %f" % trainingAccuracy
print "Training precision, recall, fscore:"
print precision_recall_fscore_support(y_train, trainingPredictions, average="binary")

predictions = np.where(bestModel.predict(x_test) == 1., 1., 0.)

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy

print "Testing precision, recall, fscore:"
print precision_recall_fscore_support(y_test, predictions, average="binary")

accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
print "Test Accuracy: %f" % accuracy

#
# # 3. Naive Bayes ********************************************************
# print "\n\nNaive Bayes ********************************************************"
#
# smoothing = [2.5, 1.5, 1., 0.5]
# bestParams = {"params": {}, "accuracy": 0}
# configCount = 1
# numConfigs = len(smoothing)
# for smoothingTerm in smoothing:
#     print "Evaluating configuration %i of %i. (%.2f%%)" % (
#     configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
#     print "Smoothing: %.2f" % smoothingTerm
#     configCount += 1
#     splitCount = 1
#     accuracies = []
#     for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(x_train, y_train):
#         x_train_split = x_train[trainInds]
#         y_train_split = y_train[trainInds]
#         x_eval_split = x_train[evalInds]
#         y_eval_split = y_train[evalInds]
#
#         model = NaiveBayes(smoothing=smoothingTerm)
#         model.fit(x_train_split, y_train_split)
#         predictions = model.predict(x_eval_split)
#         accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
#         accuracies.append(accuracy)
#         print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
#         splitCount += 1
#     averageAccuracy = np.mean(np.array(accuracies))
#     if averageAccuracy > bestParams["accuracy"]:
#         bestParams["accuracy"] = averageAccuracy
#         bestParams["params"]["smoothing"] = smoothingTerm
#
# print "Best params for %s:" % type(model)
# print bestParams["params"]
# print "Best Average Training Accuracy:"
# print bestParams["accuracy"]
#
# bestModel = model.set_params(bestParams["params"])
# bestModel.fit(x_train, y_train)
#
# trainingPredictions = bestModel.predict(x_train)
# trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
# print "Training Accuracy: %f" % trainingAccuracy
#
# predictions = bestModel.predict(x_test)
#
# accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
# print "Test Accuracy: %f" % accuracy
#
#
#
# # 3. Bagged Forests ********************************************************
# print "\n\nBagged Forests ********************************************************"
#
# model = BaggedForest(numTrees=1000, depth=3)
# model.fit(x_train, y_train)
#
# trainingPredictions = np.where(model.predict(x_train) == 1., 1., 0.)
# trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
# print "Training Accuracy: %f" % trainingAccuracy
# print "Training precision, recall, fscore:"
# print precision_recall_fscore_support(y_train, trainingPredictions, average="binary")
#
# predictions = np.where(model.predict(x_test) == 1., 1., 0.)
# print "Testing precision, recall, fscore:"
# print precision_recall_fscore_support(y_test, predictions, average="binary")
#
# accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
# print "Test Accuracy: %f" % accuracy
#
# #pickle.dump(model, open(baggedForestModelPath, "wb"))
#
#
# # 4. SVM over Bagged Forest ********************************************************
#
# # baggedForestModel = pickle.load(open(baggedForestModelPath, "rb"))
# # baggedForestPredictionsTrain = model.predict(x_train, unaveraged=True).T
# # baggedForestPredictionsTest = model.predict(x_test, unaveraged=True).T
# # pickle.dump(baggedForestPredictionsTrain, open(baggedForestPredictionsTrainPath, "wb"))
# # pickle.dump(baggedForestPredictionsTest, open(baggedForestPredictionsTestPath, "wb"))
#
# print "\n\nSVM over Bagged Forest ********************************************************"
# baggedForestPredictionsTrain = csr_matrix(pickle.load(open(baggedForestPredictionsTrainPath, "rb")))
# baggedForestPredictionsTest = csr_matrix(pickle.load(open(baggedForestPredictionsTestPath, "rb")))
#
# svmC = [10, 1, .1, .01]
# rate = [.001, .00001]
# numEpochs = 3
# bestParams = {"params": {}, "accuracy": 0}
# configCount = 1
# numConfigs = len(svmC) * len(rate)
# for C in svmC:
#     for r in rate:
#         print "Evaluating configuration %i of %i. (%.2f%%)" % (configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
#         print "C: %.4f      r: %.2f" % (C, r)
#         configCount += 1
#         splitCount = 1
#         accuracies = []
#         for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(baggedForestPredictionsTrain, y_train):
#             x_train_split = baggedForestPredictionsTrain[trainInds]
#             y_train_split = y_train[trainInds]
#             x_eval_split = baggedForestPredictionsTrain[evalInds]
#             y_eval_split = y_train[evalInds]
#
#             model = SVM(C=C, r=r)
#             model.fit(x_train_split, y_train_split)
#             predictions = model.predict(x_eval_split)
#             accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
#             accuracies.append(accuracy)
#             print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
#             splitCount += 1
#         averageAccuracy = np.mean(np.array(accuracies))
#         if averageAccuracy > bestParams["accuracy"]:
#             bestParams["accuracy"] = averageAccuracy
#             bestParams["params"]["C"] = C
#             bestParams["params"]["r"] = r
#
#
#
# print "Best params for %s: (Over Bagged Forest)" % type(model)
# print bestParams["params"]
# print "Best Average Training Accuracy:"
# print bestParams["accuracy"]
#
# bestModel = model.set_params(bestParams["params"])
# bestModel.fit(baggedForestPredictionsTrain, y_train)
#
# trainingPredictions = bestModel.predict(baggedForestPredictionsTrain)
# trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
# print "Training Accuracy: %f" % trainingAccuracy
#
# predictions = bestModel.predict(baggedForestPredictionsTest)
#
# accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
# print "Test Accuracy: %f" % accuracy
#
# #5. Logistic Regression over Bagged Forest ********************************************************
#
# print "\n\nLogistic Regression over Bagged Forest ********************************************************"
#
# tradeoff = [10., 100., 1000., 10000.]
# rate = [.001, .0001, .000001]
# numEpochs = 3
# bestParams = {"params": {}, "accuracy": 0}
# configCount = 1
# numConfigs = len(tradeoff) * len(rate)
# for tradeoffConstant in tradeoff:
#     for r in rate:
#         print "Evaluating configuration %i of %i. (%.2f%%)" % (
#         configCount, numConfigs, float(configCount) * 100 / float(numConfigs))
#         print "Tradeoff: %.4f      r: %.2f" % (tradeoffConstant, r)
#         configCount += 1
#         splitCount = 1
#         accuracies = []
#         for trainInds, evalInds in StratifiedKFold(n_splits=5, random_state=0).split(baggedForestPredictionsTrain, y_train):
#             x_train_split = baggedForestPredictionsTrain[trainInds]
#             y_train_split = y_train[trainInds]
#             x_eval_split = baggedForestPredictionsTrain[evalInds]
#             y_eval_split = y_train[evalInds]
#
#             model = LogisticRegression(sigma=tradeoffConstant, r=r)
#             model.fit(x_train_split, y_train_split)
#             predictions = model.predict(x_eval_split)
#             accuracy = float(np.sum(np.where(predictions == y_eval_split, 1, 0))) / float(len(y_eval_split))
#             accuracies.append(accuracy)
#             print "Split %i of %i. Accuracy: %.2f" % (splitCount, 5, accuracy)
#             splitCount += 1
#         averageAccuracy = np.mean(np.array(accuracies))
#         if averageAccuracy > bestParams["accuracy"]:
#             bestParams["accuracy"] = averageAccuracy
#             bestParams["params"]["sigma"] = tradeoffConstant
#             bestParams["params"]["r"] = r
#
# print "Best params for %s:" % type(model)
# print bestParams["params"]
# print "Best Average Training Accuracy:"
# print bestParams["accuracy"]
#
# bestModel = model.set_params(bestParams["params"])
# bestModel.fit(baggedForestPredictionsTrain, y_train)
#
# trainingPredictions = bestModel.predict(baggedForestPredictionsTrain)
# trainingAccuracy = np.sum(np.where(trainingPredictions == y_train, 1., 0.)) / float(len(y_train))
# print "Training Accuracy: %f" % trainingAccuracy
#
# predictions = bestModel.predict(baggedForestPredictionsTest)
#
# accuracy = np.sum(np.where(predictions == y_test, 1., 0.)) / float(len(y_test))
# print "Test Accuracy: %f" % accuracy