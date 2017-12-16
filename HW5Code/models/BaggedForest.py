import numpy as np
from DecisionTree import DecisionTree
import sys
import pickle


class BaggedForest(object):
    def __init__(self, numTrees=1000, depth=3):
        self.numTrees = numTrees
        self.depth = depth
        self.trees = []
        self.indices = []

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def fit(self, X, Y):
        self.trees = []
        Y = np.where(Y==1., 1., 0.)
        X = X.tocsc()
        np.random.seed(0)
        featureNames = [str(index) for index in range(100)]
        allIndices = list(range(X.shape[1]))
        for index in range(self.numTrees):
            sys.stdout.write("\rFitting tree %i of %i. (%.2f%%)" % (index + 1, 1000, float(index + 1) * 100. / float(1000)))
            sys.stdout.flush()
            indices = np.random.choice(allIndices, 100)
            currentX = X[:, indices].toarray()
            currentTree = DecisionTree(maxDepth=self.depth)
            currentTree.fit(currentX, Y, featureNames)
            self.trees.append(currentTree)
            self.indices.append(indices)
        # Erase to end of line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


    def predict(self, X, unaveraged=False):
        # REMEMBER to transform the output Y to have labels in {1, -1} not {1, 0}.
        X = X.tocsc()
        predictions = np.zeros((self.numTrees, X.shape[0]))
        featureNames = [str(index) for index in range(100)]
        for index in range(len(self.trees)):
            indices = self.indices[index]
            tree = self.trees[index]
            currentX = X[:, indices].toarray()
            currentPredictions = tree.predict(currentX, featureNames)
            predictions[index] = currentPredictions

        votedPredictions = np.mean(predictions, axis=0)
        if unaveraged:
            votedPredictions = predictions
        finalPredictions = np.where(votedPredictions > .05, 1., -1.)

        return finalPredictions