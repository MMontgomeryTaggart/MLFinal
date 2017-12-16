import numpy as np
from scipy.sparse import csr_matrix
import sys
import math


class NaiveBayes(object):
    def __init__(self, smoothing=1.):
        self.aPos = None
        self.bPos = None
        self.aNeg = None
        self.bNeg = None
        self.PPos = None
        self.PNeg = None
        self.smoothing = smoothing

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def fit(self, X, Y):
        numFeatures = X.shape[1]

        Ppos = np.sum(np.where(Y == 1, 1., 0.)) / float(len(Y))
        Pneg = 1. - Ppos
        self.PPos = math.log(Ppos, 2)
        self.PNeg = math.log(Pneg, 2)

        self.aPos = np.zeros(numFeatures)
        self.bPos = np.zeros(numFeatures)

        S = 2.

        YPos = csr_matrix(np.expand_dims(np.where(Y == 1., 1., 0.), axis=1))
        YNeg = csr_matrix(np.expand_dims(np.where(Y == -1., 1., 0.), axis=1))

        aDenom = np.sum(np.where(Y==1, 1, 0)) + (S * self.smoothing)
        bDenom = np.sum(np.where(Y == -1, 1, 0)) + (S * self.smoothing)

        XPosPos = X.multiply(YPos)
        XPosNeg = X.multiply(YNeg)

        aNum = np.squeeze(np.array(XPosPos.sum(axis=0))) + (S * self.smoothing)
        bNum = np.squeeze(np.array(XPosNeg.sum(axis=0))) + (S * self.smoothing)

        self.aPos = np.divide(aNum, aDenom)
        self.bPos = np.divide(bNum, bDenom)

        self.aNeg = 1. - self.aPos
        self.bNeg = 1. - self.bPos

        self.aPos = np.log2(self.aPos)
        self.bPos = np.log2(self.bPos)
        self.aNeg = np.log2(self.aNeg)
        self.bNeg = np.log2(self.bNeg)



    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        for index in range(X.shape[0]):
            row = X[index].toarray()
            postPosProbs = np.where(row == 1., self.aPos, self.aNeg)
            postNegProbs = np.where(row == 1., self.bPos, self.bNeg)

            sumPostPos = np.sum(postPosProbs)
            sumPostNeg = np.sum(postNegProbs)

            probPos = sumPostPos + self.PPos
            probNeg = sumPostNeg + self.PNeg
            if probPos > probNeg:
                predictions[index] = 1.
            else:
                predictions[index] = -1.
        return predictions