import numpy as np

class LogisticActivation(object):

    @classmethod
    def calculateActivation(cls, inputArray):
        output = np.array(inputArray, dtype=np.float32)
        output = np.exp(-output)
        output = 1. / (1. + output)
        return output

    @classmethod
    def derivative(cls, inputArray):
        output = cls.calculateActivation(inputArray)
        return np.multiply(output, (1. - output))




class ReluActivation(object):

    @classmethod
    def calculateActivation(cls, inputArray):
        output = np.array(inputArray, dtype=np.float32)
        return np.where(output < 0., 0., output)

    @classmethod
    def derivative(cls, inputArray):
        output = cls.calculateActivation(inputArray)
        return np.where(output < 0., 0., 1.)

# class LinearActivation(object):
#
#     @classmethod
#     def calculateActivation(cls, inputArray):
#         return inputArray


