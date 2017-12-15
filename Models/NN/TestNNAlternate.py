from FeedForwardNetwork import FeedForwardNet
from Losses import linearLoss, squaredLoss
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from Activations import ReluActivation, LogisticActivation
import pickle
import pandas as pd
import sys

TRAIN_CORPUS_PATH = "../../Data/Corpus_Train/"
TRAIN_TRUTH_PATH = "../../Data/GoldStandard/TrainingDocumentClasses.txt"

TEST_CORPUS_PATH = "../../Data/Corpus_Test/"
TEST_TRUTH_PATH = "../../Data/GoldStandard/TestDocumentClasses.txt"

PICKLE_DATA_PATH = "../../Data/BOWData.pkl"
TEST_PICKLE_DATA_PATH = "../../Data/TestBOWData.pkl"

def specificity(true, predicted):
    numTrueNegatives = np.sum(np.where(true == 0, 1., 0.))
    numAgreedNegatives = np.sum(((predicted - 1) * -1) * ((true - 1) * -1))
    return float(numAgreedNegatives) / float(numTrueNegatives)

def NPV(true, predicted):
    true = (true - 1) * -1
    predicted = (predicted - 1) * -1
    return sklearn.metrics.precision_score(true, predicted)

def printScores(true, predicted):
    precision, recall, fscore, _ = precision_recall_fscore_support(true, predicted, average="binary")
    accuracy = accuracy_score(true, predicted)
    specificity_score = specificity(true, predicted)
    npv = NPV(true, predicted)

    print("Accuracy: %.3f\nF-Score: %.3f\nPrecision: %.3f\nRecall (Sensitivity): %.3f\nSpecificity: %.3f\nNPV: %.3f" % (accuracy, fscore, precision, recall, specificity_score, npv))


#Unpickle the data
data = pickle.load(open(PICKLE_DATA_PATH, 'rb'))
testData = pickle.load(open(TEST_PICKLE_DATA_PATH, 'rb'))

#Set up train, validation, test data
validationFraction = .2
np.random.seed(0)

X_train_all = np.asarray(data["features"])
Y_train_all = data["labels"]

shuffledIndices = np.arange(X_train_all.shape[0])
np.random.shuffle(shuffledIndices)
X_train_all = X_train_all[shuffledIndices]
Y_train_all = Y_train_all[shuffledIndices]

numTraining = int(X_train_all.shape[0] * (1 - validationFraction))
X_train = X_train_all[:numTraining]
Y_train = Y_train_all[:numTraining]
X_validation = X_train_all[numTraining:]
Y_validation = Y_train_all[numTraining:]

X_test = testData["features"]
Y_test = testData["labels"]


learningRateList = [0.1, 0.01, 0.001, 0.0001, 0.00001]
numEpochsList = [1, 10, 100, 500, 1000, 3000]

results = {"learningRate" : [], "numEpochs" : [], "fscore" : [], "precision" : [], "recall" : []}

totalConfigs = len(learningRateList) * len(numEpochsList)
count = 1

for learningRate in learningRateList:
    for numEpochs in numEpochsList:
        sys.stdout.write("\rRunning Configuration %i of %i. %.2f%% complete" % (
            count, totalConfigs, float(count) * 100. / float(totalConfigs)))
        count += 1
        model = FeedForwardNet(X_train.shape[1], 1, activation=LogisticActivation, hiddenLayerSizes=[25], batchSize=1, epochs=numEpochs, lossFunction=linearLoss, randomSeed=0, learningRate=learningRate)
        model.fit(X_train, Y_train)
        rawPredictions = model.predict(X_validation)
        predictions = np.where(rawPredictions > .5, 1., 0.)
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_validation, predictions, average="binary")
        results["learningRate"].append(learningRate)
        results["numEpochs"].append(numEpochs)
        results["fscore"].append(fscore)
        results["precision"].append(precision)
        results["recall"].append(recall)



dataFrame = pd.DataFrame(results)
dataFrame.to_csv("./GraphOutput/Rates_LogisticActivation_1Hidden_25Units_30Points.tsv", index=False, sep='\t')

# rawPredictions = model.predict(X)#, shape=(X.shape[0], 2))
# predictions = np.where(rawPredictions > .5, 1., 0.)
#
# printScores(Y, predictions)
# # print predictions[0] + predictions[4]
# # print predictions[1] + predictions[5]
# # print predictions[2] + predictions[6]
# # print predictions[3] + predictions[7]
# print np.sum(predictions)













# X = np.array([[-10., -10.], [-10., -10.], [-10., -10.], [-10., -10.]])#, [-4., -5.]])#, [-4.5, -6.]])#, [-3., -8.]])#, [-5., -2.]])
# Y = np.array([0., 0., 0., 0.])

# dataset = np.array([[2.7810836, 2.550537003, 0],
#            [1.465489372, 2.362125076, 0],
#            [3.396561688, 4.400293529, 0],
#            [1.38807019, 1.850220317, 0],
#            [3.06407232, 3.005305973, 0],
#            [7.627531214, 2.759262235, 1],
#            [5.332441248, 2.088626775, 1],
#            [6.922596716, 1.77106367, 1],
#            [8.675418651, -0.242068655, 1],
#            [7.673756466, 3.508563011, 1]])
#
# X = dataset[:, [0,1]]
# Ycolumn = dataset[:, 2]
# Y = np.zeros((len(Ycolumn), 2))
# Y[:, 1] = Ycolumn
# Y[:, 0] = 1 - Ycolumn

# xpos = np.random.normal(20, 4, 100)
# xneg = np.random.normal(-5, 4, 100)
# ypos = np.random.normal(10, 4, 100)
# Yneg = np.random.normal(-5, 4, 100)
# x = np.concatenate(xpos, xneg)
# y = np.expand_dims(np.concatentate(ypos, yneg),
#
# X = np.concatenate(x, y, axis=)