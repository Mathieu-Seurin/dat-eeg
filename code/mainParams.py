#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

import argparse

class ArgumentError(Exception): pass
    
parser = argparse.ArgumentParser()
allDataType = ['test','random','r','s','f','fs','patch']
allModelType = ['lin','nonLin']

#=========================  PARAMETERS  ============================
#===================================================================

#Needed Parameters
parser.add_argument("data", help="Type of Data you want to learn/transform",choices=allDataType)
parser.add_argument("model", help="Type of Model you want to learn/transform",choices=allModelType)

#Optionnal Argument
parser.add_argument("-s", "--subject", help="Which subject to use (default : A)", choices=['A','B','AB','BA'], default='A')

parser.add_argument("-j","--jobs",help="Number of jobs used to learn the data", type=int, default=1)

parser.add_argument("--cardRandom",help="Number of features for random data", type=int, default=2)

parser.add_argument("-i","--freqMin", help="Frequency filter's lower bound", type=float, default=0.1)
parser.add_argument("-a","--freqMax", help="Frequency filter's upper bound", type=float, default=60)
parser.add_argument("-d","--decimation", help="Decimation Factor (Downsampling)", type=int, default=4)

parser.add_argument("-w","--sizeWindow", help="Size of STFT window", type=float, default=0.2)

parser.add_argument("--scoring", help="Score Function used for CV (f1, roc_auc, accuracy)", choices=['f1', 'roc_auc', 'accuracy'], default='f1')

parser.add_argument("--ratioTest", help="Ratio Train/Test used (default : 0, no Test)", type=float, default=0)

parser.add_argument("--cardPatch", help="Number of Patch you want to extract", type=int, default=10000)
parser.add_argument("--LDAcompress", help="Size of features compression (default : 2)", type = int, default=2)

parser.add_argument("-c", "--copyResults",action="store_true") 
parser.add_argument("-t", "--transfer", help="Transfer Learning if indicated", action="store_true") 


args = parser.parse_args()
print(args,'\n')

data = args.data.lower()
model = args.model.lower()
#=========================  DATA  ==================================
#===================================================================
if data == 'test':

    dataType = 'raw'
    X = np.array([[i for i in range(64*4)] for j in range(50)])
    X = np.concatenate((X, -X))

    y = np.array([1 for j in range(50)])
    y = np.concatenate((y,-y))

    xTest = []
    yTest = []

elif data == 'random':
    X = np.random.random((15300,args.cardRandom))
    y = np.load(PATH_TO_DATA+'AfullY.npy')

    xTest = []
    yTest = []
    dataType = 'random'

elif data == 'r':
    
    X,y,xTest,yTest = prepareRaw(args.subject,args.ratioTest)
    dataType='raw'

elif data == 's':

    X,y,xTest,yTest = prepareStft(args.subject,args.sizeWindow,args.ratioTest)
    dataType="stft{}".format(args.sizeWindow)

elif data=='f':

    subject=args.subject
    freqMin=args.freqMin
    freqMax = args.freqMax
    decimation = args.decimation
    
    X,y,xTest,yTest = prepareFiltered(subject,freqMin,freqMax,decimation,args.ratioTest)
    dataType = '{}filtered{}{}{}'.format(subject,freqMin,freqMax,decimation)

elif data == 'fs':

    subject=args.subject
    freqMin=args.freqMin
    freqMax = args.freqMax
    decimation = args.decimation
    sizeWindow = args.sizeWindow
    
    X,y,xTest,yTest = prepareFilteredStft(subject, freqMin, freqMax, decimation,frameSize, args.ratioTest)
    datatype = '{}filtered{}{}{}Stft{}'.format(subject,freqMin,freqMax,decimation,sizeWindow)

elif data == 'patch':

    cardPatch = args.cardPatch
    
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

print(dataType)

if args.transfer:

    X,y,xTest,yTest = prepareTransfertFiltered(0.5,30,4)
    dataType = 'filtered4'

    print("+ Transfer")

#====================================================================
#=========================  MODEL  ==================================
#====================================================================
#====================================================================
cardJobs=args.jobs


if model=='lin':
    
    print(X.shape)
    learnHyperLinear(X, y, xTest, yTest, 'l2', 'f1',transformedData=dataType,jobs=cardJobs)

elif model=='nonLin':
    
    learnHyperNonLinear(X, y, xTest, yTest, 'roc_auc',transformedData=dataType,jobs=cardJobs)

elif model == 'elastic' :

    learnElasticNet(X,y,xTest,yTest, 'roc_auc', transformedData=dataType, jobs=cardJobs)
    
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

