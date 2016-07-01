#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

import argparse

class ArgumentError(Exception): pass
    
parser = argparse.ArgumentParser()


#Needed Parameters
parser.add_argument("data", help="Type of Data you want to learn/transform")
parser.add_argument("model", help="Type of Model you want to learn/transform")

#Optionnal Argument
parser.add_argument("-s", "--subject", help="Which subject to use (default : A)", choices=['A','B','AB','BA'], default='A')

parser.add_argument("-j","--jobs",help="Number of jobs used to learn the data", type=int, default=1)

parser.add_argument("--cardRandom",help="Number of features for random data", type=int, default=2)

parser.add_argument("--FreqMin", help="Frequencies' filter's lower bound", type=float)
parser.add_argument("--FreqMax", help="Frequencies' filter's upper bound", type=float)
parser.add_argument("--decimation", help="Decimation Factor (Downsampling)", type=int)

parser.add_argument("--sizeWindow", help="Size of STFT window", type=float, default=0.2)

parser.add_argument("--scoring", help="Score Function used for CV (f1, roc_auc, accuracy)", choices=['f1', 'roc_auc', 'accuracy'], default='f1')

parser.add_argument("--ratioTest", help="Ratio Train/Test used (default : 0, no Test)", type=float, default=0)

parser.add_argument("--cardPatch", help="Number of Patch you want to extract", type=int, default=10000)
parser.add_argument("--LDAcompress", help="Size of features compression (default : 2)", type = int)

parser.add_argument("-c", "--copyResults",action="store_true") 


args = parser.parse_args()
print(args)

data = parser.data

#=========================  DATA  ==================================
#===================================================================
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
    X = np.random.random((15300,10000))
    y = np.load(PATH_TO_DATA+'AfullY.npy')

    xTest = []
    yTest = []
    dataType = 'random'

elif data == 'raw':
    
    X,y,xTest,yTest = prepareRaw(subject=args.subject,splitTrainTest=args.ratioTest)
    dataType='raw'

elif data == 'stft':

    X,y,xTest,yTest = prepareStft(subject=args.subject,args.sizeWindow)
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


elif data == 'patch':
    cardPatch = 10000

    if os.path.exists('{}patchedMean{}.npy'.format(PATH_TO_DATA,cardPatch)):
        print("Loading Model...")
        X = np.load('{}patchedMean{}.npy'.format(PATH_TO_DATA,cardPatch))
    else:
        X = np.load(PATH_TO_DATA+'AfullFiltered4StftMatrix0.2X.npy')
        patcher = Patcher(X,'mean',cardPatch)

        a = time.time()
        X = patcher.patchFeatures()
        print(X.shape)
        print(time.time()-a)

    y = np.load(PATH_TO_DATA+'AfullY.npy')
    xTest = []
    yTest = []
    dataType = 'patchedMean{}'.format(cardPatch)

elif data == 'LDA':

    cardPatch = 2
    X = np.load('{}patchedLDA{}.npy'.format(PATH_TO_DATA,cardPatch))
    y = np.load(PATH_TO_DATA+'AfullY.npy')
    xTest = []
    yTest = []
    dataType = 'patchedLDA{}'.format(cardPatch)
    
    
else :
    print(USAGE)
    raise ArgumentError("Wrong Type of Data")

#====================================================================
#=========================  MODEL  ==================================
#====================================================================
#====================================================================

if model=='lin':
    
    print(X.shape)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'f1',transformedData=dataType,jobs=4)

elif model=='nonLin':
    
    learnHyperNonLinear(X, y, xTest, yTest, 'roc_auc',transformedData=dataType,jobs=4)

elif model == 'elastic' :

    learnElasticNet(X,y,xTest,yTest, 'roc_auc', transformedData=dataType, jobs=2)
    
elif model=='LDA':

    learnLDA(X,y,xTest,yTest,transformedData=dataType,n_components=100)
        
elif model=='single':
    
    clf = svm.LinearSVC(C=1e-5, class_weight='balanced')
    clf.fit(X,y)
    yPredTrain = clf.predict(X)
    yPredTest = clf.predict(xTest)

    scores = getScores(y, yPredTrain, yTest, yPredTest)

elif model == 'allS':

    learnHyperLinear(X, y, xTest, yTest, 'l2', 'accuracy',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'precision',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'recall',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'f1',transformedData=dataType)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'roc_auc',transformedData=dataType)

elif model == 'step':
    learnStep(X, y, xTest, yTest, 'l2', 'roc_auc', transformedData=dataType,jobs=3)

elif model == 'elec':
    learnElecFaster(X, y, xTest, yTest, 'l2', 'roc_auc', transformedData=dataType,jobs=4)
    
else :
    print(USAGE)
    raise ArgumentError("Wrong Type of Model")

if copyResults:
    os.system("cp Results/*.txt ~")

