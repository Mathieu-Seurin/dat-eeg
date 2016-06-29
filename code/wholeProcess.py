#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

class ArgumentError(Exception): pass
    
USAGE = """USAGE :
    python3 wholeProcess.py dataType modelType copyResults

    dataType can be : test, AR (subject A, raw) , AS (A, STFT), AF, AFS, ABF 
    modelType can be : lin, nonLin, elastic, single, allS (compute linearModel for all scores)

    if argument copyResults is specified : copy all resulst to your $HOME ~
    else does nothing
    The last argument can be ABSOLUTELY anything.

    Exemple :
    
    python3 wholeProcess.py test nonLin 42
    >> 'test data, nonLin Model, copy Results at the end'

    """

if len(sys.argv) < 2:
    print(USAGE)
    raise ArgumentError('Missing 2 arguments')
    
elif len(sys.argv) < 3:
    print(USAGE)
    raise ArgumentError('Missing 1 argument')

    
else:
    data = sys.argv[1]
    params = sys.argv[2]
    copyResults = len(sys.argv) > 3

    toPrint = "{} data, {} Model, ".format(data, params)
    if not copyResults: toPrint += "don't "
    toPrint += "copy Results at the end"

    print(toPrint)

#=========================  DATA  ==================================
if data == 'test' :

    dataType='raw'

    party = np.array([i for i in range(30)])
    X = np.concatenate((np.array([[-i,-i] for i in range(1,30)]), -np.array([[-i,-i] for i in range(1,30)]), np.array([[-i,-i] for i in range(40,80)]), -np.array([[-i,-i] for i in range(40,80)])))
    y = np.concatenate((np.array([1 for i in range(1,30)]), -np.array([1 for i in range(1,30)]),\
                        np.array([1 for i in range(40,80)]), -np.array([1 for i in range(40,80)])))

    xTest = []
    yTest = []

elif data == 'test2':

    dataType = 'raw'
    X = np.array([[i for i in range(64*4)] for j in range(50)])
    X = np.concatenate((X, -X))

    y = np.array([1 for j in range(50)])
    y = np.concatenate((y,-y))

    xTest = []
    yTest = []

elif data == 'random':
    X = np.random.random((15300,2000))
    y = np.load(PATH_TO_DATA+'AfullY.npy')

    xTest = []
    yTest = []
    dataType = 'random'
    
elif data =='AR':

    X,y,xTest,yTest = prepareRaw('A')
    dataType='raw'

elif data=='AS':

    X,y,xTest,yTest = prepareStft('A', 0.2)
    dataType='Stft'+str(frameSize)

elif data=='ABR':
    
    X,y,xTest,yTest = prepareRawAB()    
    dataType='raw'
    
elif data=='AF':

    X,y,xTest,yTest = prepareFiltered('A',0.5,30,4)
    dataType = 'filtered4'

elif data=='BF':
    X,y,xTest,yTest = prepareFiltered('B',0.5,30,4)
    dataType = 'filtered4'

elif data=='ABF':

    X,y,xTest,yTest = prepareFilteredAB(0.5,30,4)
    dataType = 'filtered4'

elif data=='AF8':

    X,y,xTest,yTest = prepareFiltered('A',0.5,15,8)
    dataType = 'filtered8'

elif data=='AFS':

    X,y,xTest,yTest = prepareFilteredStft('A',0.5,10,8,0.1)
    dataType = 'filtered4Stft0.1'

elif data=='AF8S':

    X,y,xTest,yTest = prepareFilteredStft('A',0.5,30,8, 0.1)
    dataType = 'filtered8Stft0.1'

elif data=='BFS':

    X,y,xTest,yTest = prepareFilteredStft('B',0.5,30,4, 0.1)
    dataType = 'filtered4Stft0.1'

elif data=='BF8S':

    X,y,xTest,yTest = prepareFilteredStft('B',0.5,30,8, 0.1)
    dataType = 'filtered8Stft0.1'

elif data=='trans':

    X,y,xTest,yTest = prepareTransfertFiltered(0.5,30,4)
    dataType = 'filtered4'

else :
    print(USAGE)
    raise ArgumentError("Wrong Type of Data")

#====================================================================
#=========================  MODEL  ==================================
#====================================================================
#====================================================================

if params=='lin':
    
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'f1',transformedData=dataType,jobs=4)

elif params=='nonLin':
    
    learnHyperNonLinear(X, y, xTest, yTest, 'roc_auc',transformedData=dataType,jobs=4)

elif params == 'elastic' :

    learnElasticNet(X,y,xTest,yTest, 'roc_auc', transformedData=dataType, jobs=2)
    
elif params=='single':
    
    clf = svm.LinearSVC(C=1e-5, class_weight='balanced')
    clf.fit(X,y)
    yPredTrain = clf.predict(X)
    yPredTest = clf.predict(xTest)

    scores = getScores(y, yPredTrain, yTest, yPredTest)

elif params == 'allS':

    learnHyperLinear(X, y, xTest, yTest, 'l2', 'accuracy',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'precision',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'recall',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'f1',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'roc_auc',transformedData=dataType)

elif params == 'step':
    learnStep(X, y, xTest, yTest, 'l2', 'roc_auc', transformedData=dataType,jobs=3)

elif params == 'elec':
    learnElecFaster(X, y, xTest, yTest, 'l2', 'roc_auc', transformedData=dataType,jobs=4)
    
else :
    print(USAGE)
    raise ArgumentError("Wrong Type of Model")

if copyResults:
    os.system("cp Results/*.txt ~")
