import numpy as np


def squaredLoss(predicted, true):
    difference = predicted - true
    return np.power(difference, 2.)

def linearLoss(predicted, true):
    return true - predicted