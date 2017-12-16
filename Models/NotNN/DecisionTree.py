import numpy as np
import math

def CalculateAverageEntropySingleColumn(featureColumn, labelColumn):
    uniqueValues = np.unique(featureColumn)
    entropyTerms = np.zeros(len(uniqueValues))
    for index, value in enumerate(uniqueValues):
        indices = np.where(featureColumn == value)
        featureRows = featureColumn[indices]
        labelRows = labelColumn[indices]
        valueCount = len(featureRows)
        equalResult = np.where(featureRows == labelRows)
        matchCount = len(equalResult[0])
        frequency = float(matchCount) / float(valueCount)
        if frequency == 0 or frequency == 1:
            entropyTerms[index] = 0
        else:
            entropyTerms[index] = -frequency * math.log(frequency, 2) - ((1.-frequency) * math.log(1.-frequency, 2))

    return np.mean(entropyTerms)


def CalculateAverageEntropies(matrix):
    """Calculates the entropy for each column and returns as vector of values."""
    averageEntropies = np.zeros(np.shape(matrix)[1] - 1)
    for i in range(np.shape(matrix)[1] - 1):
        averageEntropies[i] = CalculateAverageEntropySingleColumn(matrix[:, i], matrix[:, -1])
    return averageEntropies

def FindHighestInformationGain(matrix, features):
    """Returns the label of the column with the highest information gain."""
    labelEntropy = CalculateLabelEntropy(matrix[:, -1])
    columnEntropies = CalculateAverageEntropies(matrix)

    diff = labelEntropy - columnEntropies
    maxIndex = np.argmax(diff)
    return features[maxIndex]


def calculateSingleEntropyTerm(frequency):
    if frequency == 0.:
        return 0
    else:
        return -frequency * math.log(frequency, 2)

def CalculateLabelEntropy(labelVec):
    labelVec = labelVec.astype(int)
    counts = np.bincount(labelVec)
    totalLabels = len(labelVec)
    frequencies = np.divide(counts.astype(float), totalLabels)
    individualEntropyTerms = map(calculateSingleEntropyTerm, frequencies)
    entropy = np.sum(individualEntropyTerms)
    return entropy

def RemoveFeature(matrix, features, featureToRemove):
    featureIndex = features.index(featureToRemove)
    skinnierMatrix = np.delete(matrix, featureIndex, 1)
    skinnierFeatures = list(features)
    del skinnierFeatures[featureIndex]
    return skinnierMatrix, skinnierFeatures

def FilterRows(matrix, features, feature, value):
    """Returns all rows with the value 'value' in the column with the label 'feature'."""
    featureIndex = features.index(feature)
    featureColumn = matrix[:, featureIndex]
    indices = np.where(featureColumn == value)
    rows = np.squeeze(matrix[indices, :])
    # if after squeezing we are down to 1-d, add a second dimension
    if len(rows.shape) == 1:
        rows = np.expand_dims(rows, axis=0)

    return rows

class Node:
    def __init__(self, featureName, value, level):
        self.feature = featureName
        self.value = value  #The value of the attribute that the parent node split on to create this node.
        self.children = []
        self.level = level

    def addChild(self, child):
        """
        Child may be a node or a leaf.
        """
        self.children.append(child)

class Leaf:
    def __init__(self, featureValue, label, level):
        self.value = featureValue
        self.label = label
        self.level = level


class DecisionTree(object):
    def __init__(self, maxDepth=None):
        self.tree = None
        self.maxDepth = maxDepth

    def fit(self, X, Y, features):
        # Append the labels onto the features
        Y = np.expand_dims(Y, axis=1)
        matrix = np.concatenate((X, Y), axis=1)
        # perform initial step to get the root node
        # Base case 1: currentMatrix has only one column, the label column. Return the majority label.
        if np.shape(matrix)[1] == 1:
            labelColumn = np.squeeze(matrix.T)
            counts = np.bincount(labelColumn)
            return Leaf(None, np.argmax(counts), 1)

        # Base case 2: currentMatrix has more than one column but all the labels have the same value. Return the value.
        labelColumn = np.squeeze(matrix[:, -1].T)
        if len(np.unique(labelColumn)) == 1:
            return Leaf(None, labelColumn[0], 1)

        # if maxDepth is 1, return the majority label:
        if self.maxDepth == 1:
            labelColumn = np.squeeze(matrix[:, -1].T)
            counts = np.bincount(labelColumn)
            return Leaf(0, np.argmax(counts), 1)


        # Split on the column with the highest information gain, create a new node for one of the values of that feature,
        # call induceTree on the new node, and return the node.
        featureToSplit = FindHighestInformationGain(matrix, features)
        indexOfFeatureToSplit = features.index(featureToSplit)
        uniqueFeatureValues = np.unique(matrix[:, indexOfFeatureToSplit])

        rootNode = Node(featureToSplit, None, 1)

        for value in uniqueFeatureValues:
            self.induceTree(rootNode, matrix, features, featureToSplit, value)

        self.tree = rootNode

        # grow the rest of the tree

    def induceTree(self, parentNode, currentMatrix, currentFeatures, featureToRemove, valueToSplit):
        # Remove the column we are splitting on and grab only the rows that match 'valueToSplit', examine result for base case situations, if none, call induce tree with
        # reduced matrix and features after adding a child node to the parent
        rowSubset = FilterRows(currentMatrix, currentFeatures, featureToRemove, valueToSplit)
        skinnyRowSubset, skinnierFeatures = RemoveFeature(rowSubset, currentFeatures, featureToRemove)
        currentLevel = parentNode.level + 1


        # Base case 1: currentMatrix has only one column, the label column. Return the majority label.
        if np.shape(skinnyRowSubset)[1] == 1:
            labelColumn = np.squeeze(skinnyRowSubset.T).astype(int)
            if len(labelColumn.shape) < 1:
                labelColumn = np.expand_dims(labelColumn, axis=0)
            counts = np.bincount(labelColumn)
            parentNode.addChild(Leaf(valueToSplit, np.argmax(counts), currentLevel))
            return parentNode


        # Base case 2: currentMatrix has more than one column but all the labels have the same value. Return the value.
        labelColumn = np.squeeze(skinnyRowSubset[:, -1].T).astype(int)
        # make sure the label column is still a list so we can index it.
        if len(labelColumn.shape) == 0:
            labelColumn = np.expand_dims(labelColumn, axis=0)
        if len(np.unique(labelColumn)) == 1:
            parentNode.addChild(Leaf(valueToSplit, labelColumn[0], currentLevel))
            return parentNode


        #Check if we are at the max depth. If so, return a leaf node with the value of the majority of the labels.
        if self.maxDepth:
            if self.maxDepth == currentLevel:
                labelColumn = np.squeeze(skinnyRowSubset[:, -1].T).astype(int)
                counts = np.bincount(labelColumn)
                parentNode.addChild(Leaf(valueToSplit, np.argmax(counts), currentLevel))
                return parentNode

        # Split on the column with the highest information gain, create a new node for one of the values of that feature,
            # call induceTree on the new node, and return the node.
        featureToSplit = FindHighestInformationGain(skinnyRowSubset, skinnierFeatures)
        indexOfFeatureToSplit = skinnierFeatures.index(featureToSplit)
        uniqueFeatureValues = np.unique(skinnyRowSubset[:, indexOfFeatureToSplit])

        childNode = Node(featureToSplit, valueToSplit, currentLevel)
        parentNode.addChild(childNode)

        for value in uniqueFeatureValues:
            self.induceTree(childNode, skinnyRowSubset, skinnierFeatures, featureToSplit, value)

        return parentNode

    def classifyTestExample(self, node, featureVec, featureNames):
        if isinstance(node, Leaf):
            return node.label
        featureToSplit = node.feature
        indexOfFeature = featureNames.index(featureToSplit)
        featureValue = featureVec[indexOfFeature]

        for child in node.children:
            if child.value == featureValue:
                if isinstance(child, Leaf):
                    return child.label
                else:
                    return self.classifyTestExample(child, featureVec, featureNames)

        # did not find the feature value, which means we never saw it in the training set. Return the leaf value, if there
        # is one.
        for child in node.children:
            if isinstance(child, Leaf):
                return child.label
        raise RuntimeError("This should never happen... (this means the algorithm didn't find a child whose value matched the test vector feature value.)")

    def findDeepestLeaf(self, current=0, best=0):
        current = current + 1
        if isinstance(self.tree, Leaf):
            return current

        for child in self.tree.children:
            depth = self.findDeepestLeaf(child, current, best)
            if depth > best:
                best = depth

        return best

    def predict(self, X, featureNames):
        numRows = X.shape[0]
        predictions = np.zeros(numRows)

        for index in range(numRows):
            currentRow = X[index, :]
            predictions[index] = self.classifyTestExample(self.tree, currentRow, featureNames)

        return predictions
