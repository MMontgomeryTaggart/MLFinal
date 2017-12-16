import numpy as np
import sys
import Activations
from Losses import squaredLoss

class HiddenLayer(object):
    def __init__(self, numNeurons, numInputs, activation, layerNumber):
        self.numNeurons = numNeurons
        self.activation = activation
        self.weights = np.random.uniform(0., 1., size=(numNeurons, numInputs + 1)) # +1 for bias
        self.output = np.zeros((numNeurons,))
        self.deltas = np.zeros((numNeurons,))
        self.inputs = np.zeros((numInputs + 1,)) # +1 for bias
        self.layerNumber = layerNumber

    def feedForward(self, inputs):
        inputs = np.concatenate((inputs, [1.]))
        if inputs.shape != self.inputs.shape:
            raise RuntimeError("Shape of input values does not match the expected shape (layer %i). Expected: %s, Got: %s" % (self.layerNumber, self.inputs.shape, inputs.shape))
        self.inputs = inputs
        summedWeights = np.matmul(self.weights, inputs)
        self.output = self.activation.calculateActivation(summedWeights)
        return self.output

    def assignDeltas(self, nextLayer):
        nextLayerNoBias = nextLayer.weights[:,:-1]
        errors = np.matmul(nextLayerNoBias.T, nextLayer.deltas) # Exclude the bias weights since the bias node does not propagate error.
        if errors.shape != self.deltas.shape:
            raise RuntimeError("Shape of error values does not match the expected shape (layer %i). Expected: %s, Got: %s" % (self.layerNumber, self.deltas.shape, errors.shape))
        derivativeTerm = self.activation.derivative(np.matmul(self.weights, self.inputs))
        self.deltas = np.multiply(errors, derivativeTerm)

    def adjustWeights(self, learningRate):
        # Gradient_for_one_weight = that_weight's_input * delta_value_of_receiving_neuron * learning_rate
        gradients = np.outer(self.deltas, self.inputs) * learningRate # inner product of inputs and deltas, yields 2x3 to match weights.
        adjustedWeights = self.weights + gradients
        if adjustedWeights.shape != self.weights.shape:
            raise RuntimeError(
                "Shape of weight values does not match the expected shape (layer %i). Expected: %s, Got: %s" % (
                self.layerNumber, self.weights.shape, adjustedWeights.shape))
        self.weights = adjustedWeights


class OutputLayer(HiddenLayer):
    def __init__(self, numNeurons, numInputs, activation, layerNumber, loss=squaredLoss):
        super(OutputLayer, self).__init__(numNeurons, numInputs, activation, layerNumber)
        self.lossFunction = loss

    def assignDeltas(self, averageLosses):
        if averageLosses.shape != (self.numNeurons,):
            raise RuntimeError("Shape of delta values does not match the expected shape (layer %i). Expected: %s, Got: %s" % (self.layerNumber, str((self.numNeurons,)), averageLosses.shape))
        derivative = self.activation.derivative(np.matmul(self.weights, self.inputs))
        self.deltas = np.multiply(averageLosses, derivative)


class FeedForwardNet(object):
    def __init__(self, inputSize, outputSize, hiddenLayerSizes=None, activation=Activations.LogisticActivation,
                 lossFunction=squaredLoss, randomSeed=None, batchSize=10, epochs=10, learningRate=0.1, verbose=False):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayerSizes = hiddenLayerSizes
        self.activation = activation
        self.lossFunction = lossFunction
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.averageLosses = np.zeros((outputSize,))
        self.verbose = verbose
        self.layers = []

        if randomSeed != None:
            np.random.seed(randomSeed)

        self.initializeNetwork()

    def initializeNetwork(self):
        for index, size in enumerate(self.hiddenLayerSizes):
            if index == 0:
                firstHiddenLayer = HiddenLayer(size, self.inputSize, self.activation, index + 1)
                self.layers.append(firstHiddenLayer)
            else:
                # Add a hidden layer
                nextHiddenLayer = HiddenLayer(size, self.hiddenLayerSizes[index - 1], self.activation, index + 1)
                self.layers.append(nextHiddenLayer)


        if len(self.hiddenLayerSizes) == 0:
            # No hidden layers, just an output layer with the input size as input
            outputLayer = OutputLayer(loss=self.lossFunction, *[self.outputSize, self.inputSize, self.activation, len(self.hiddenLayerSizes) + 1])
            self.layers.append(outputLayer)
        else:
            # Append the final output layer with input size equal to final hidden size
            outputLayer = OutputLayer(loss=self.lossFunction, *[self.outputSize, self.hiddenLayerSizes[-1], self.activation, len(self.hiddenLayerSizes) + 1])
            self.layers.append(outputLayer)


    def fit(self, X, Y):
        instanceCount = 1
        numInstances = X.shape[0]
        for epoch in range(self.epochs):
            for instanceIndex in range(X.shape[0]):
                if self.verbose:
                    sys.stdout.write("\rRunning Epoch %i of %i. %.2f%% complete" % (
                    epoch + 1, self.epochs, float(instanceIndex + 1) * 100. / float(numInstances)))
                instanceCount += 1
                currentX = X[instanceIndex]
                currentY = Y[instanceIndex]
                nextInput = currentX
                for layerIndex, layer in enumerate(self.layers):
                    output = layer.feedForward(nextInput)
                    if layerIndex == len(self.layers) - 1:
                        losses = self.lossFunction(output, currentY)
                        self.averageLosses += losses
                    else:
                        nextInput = output

                if instanceCount % self.batchSize == 0:
                    # apply gradient
                    self.averageLosses = self.averageLosses / float(self.batchSize)
                    for layerIndex, layer in reversed(list(enumerate(self.layers))):
                        if layerIndex == len(self.layers) - 1:
                            layer.assignDeltas(self.averageLosses)
                        else:
                            layer.assignDeltas(self.layers[layerIndex + 1])
                    for layer in reversed(self.layers):
                        layer.adjustWeights(self.learningRate)
                    self.resetLoss()
                    #print("After row: %s\nNetwork: %s\n" % (str(currentX) + str(currentY), [weights for layer in self.layers for weights in layer.weights]))
            if self.verbose:
                print("")


    def predict(self, X, shape=None):
        predictions = np.zeros((X.shape[0]))
        if shape:
            predictions = np.zeros(shape)
        for instanceIndex in range(X.shape[0]):
            instance = X[instanceIndex]
            handoffValue = instance
            for layer in self.layers:
                handoffValue = layer.feedForward(handoffValue)
            predictions[instanceIndex] = handoffValue

        return predictions

    def resetLoss(self):
        self.averageLosses = np.zeros(self.averageLosses.shape)