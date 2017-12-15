from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pickle
import numpy as np
import nltk
import re


CORPUS_PATH = "./Data/Corpus_Train/"
GOLD_STANDARD_PATH = "./Data/Gold Standard/TrainingDocumentClasses.txt"

TEST_CORPUS_PATH = "./Data/Corpus_Test/"
TEST_TRUTH_PATH = "./Data/Gold Standard/TestDocumentClasses.txt"

def getDocsAndLabels(corpusPath, goldStandardFilepath, balanceClasses=False):
    with open(goldStandardFilepath, 'rU') as f:
        rawGold = f.read()

    goldLines = rawGold.split('\n')
    noteNames = []
    noteLabels = []
    for line in goldLines:
        if line == '':
            continue
        columns = line.split('\t')
        noteNames.append(columns[0])
        noteLabels.append(int(columns[1]))

    noteNames = np.array(noteNames)
    noteLabels = np.array(noteLabels)
    if balanceClasses:
        positiveNoteNames = noteNames[np.where(noteLabels == 1)]
        negativeNoteNames = noteNames[np.where(noteLabels == 0)]

        numPos = len(positiveNoteNames)
        selectedNegatives = np.random.choice(negativeNoteNames, size=numPos, replace=False)
        noteLabels = np.concatenate((np.ones(numPos), np.zeros(numPos)))
        noteNames = np.concatenate((positiveNoteNames, selectedNegatives))

    noteBodies = []
    for name in noteNames:
        with open(corpusPath + name + ".txt", 'rU') as f:
            noteBodies.append(f.read())

    return noteBodies, noteLabels

def tokenizer(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = [token for token in tokens if re.match(r"^[a-zA-Z]+$", token)]
    return filtered_tokens


docs, labels = getDocsAndLabels(CORPUS_PATH, GOLD_STANDARD_PATH, balanceClasses=True)

vectorizer = TfidfVectorizer(stop_words="english", tokenizer=tokenizer, ngram_range=(1, 3), min_df=.001, max_df=.5)
vectorizedText = vectorizer.fit_transform(docs)

k=100
selector = SelectKBest(chi2, k=k)

selectedFeatures = selector.fit_transform(vectorizedText, labels)

# assert selectedFeatures.shape == (len(labels), k)
#
# data = {"features" : selectedFeatures.todense(), "labels" : labels}
#
# pickle.dump(data, open("./Data/BOWData.pkl", 'wb'), protocol=2)
#
# data = pickle.load(open("./Data/BOWData.pkl", 'rb'))
#
# print data["features"].shape
# print data["labels"].shape

testDocs, testLabels = getDocsAndLabels(TEST_CORPUS_PATH, TEST_TRUTH_PATH)
vectorizedTest = vectorizer.transform(testDocs)
selectedTestFeatures = selector.transform(vectorizedTest)

testData = {"features" : selectedTestFeatures.todense(), "labels" : testLabels}
pickle.dump(testData, open("./Data/TestBOWData.pkl", 'wb'), protocol=2)