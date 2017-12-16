import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
import sys


class SVM(object):
    def __init__(self, C=1, r=.1):
        self.C = C
        self.r = r
        self.w = None
        self.epochs = 1

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    def fit(self, X, Y, epochs = 1):
        self.epochs = epochs
        numFeatures = X.shape[1]
        self.w = np.zeros(numFeatures)

        count = 0
        for epoch in range(self.epochs):
            if epoch != 0:
                X, Y = shuffle(X, Y, random_state = 0)
            numRows = X.shape[0]
            for index in range(X.shape[0]):
                count += 1
                sys.stdout.write("\rProcessing %i of %i. (%.2f%%)" % (count, numRows * self.epochs, float(count) * 100. / float(numRows * self.epochs)))
                sys.stdout.flush()
                row = X[index]
                label = Y[index]
                product = row.dot(self.w)
                loss = 1. - (product * label)
                gradient = self.w * self.r
                if loss > 0:
                    rowProduct = np.squeeze(row.multiply(self.C * label * self.r).toarray())
                    gradient = gradient - rowProduct

                self.w = self.w - gradient
        # Erase to end of line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()




    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        for index in range(X.shape[0]):
            row = X[index]
            predictions[index] = np.sign(row.dot(self.w))
        return predictions