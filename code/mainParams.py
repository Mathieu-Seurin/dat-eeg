#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

import argparse

class ArgumentError(Exception): pass
    
parser = argparse.ArgumentParser()
allDataType = ['test','random','r','s','f','fs','p']
allModelType = ['lin','nonLin','forest','csp','pipeLin','pipeRidge','ridge']

#=========================  PARAMETERS  ============================
#===================================================================

#Needed Parameters
parser.add_argument("data", help="Type of Data you want to learn/transform",choices=allDataType)
parser.add_argument("model", help="Type of Model you want to use",choices=allModelType)

#Optionnal Argument
parser.add_argument("-s", "--subject", help="Which subject to use (default : A)", choices=['A','B','C','5','Z','Y'], default='A')

parser.add_argument("-j","--jobs",help="Number of jobs used to learn the data", type=int, default=1)

parser.add_argument("--cardRandom",help="Number of features for random data", type=int, default=10)

parser.add_argument("-i","--freqMin", help="Frequency filter's lower bound", type=float, default=0.5)
parser.add_argument("-a","--freqMax", help="Frequency filter's upper bound", type=float, default=30.0)
parser.add_argument("-d","--decimation", help="Decimation Factor (Downsampling)", type=int, default=4)

parser.add_argument("-w","--sizeWindow", help="Size of STFT window", type=float, default=0.2)

parser.add_argument("--scoring", help="Score Function used for CV (f1, roc_auc, accuracy)", choices=['f1', 'roc_auc', 'accuracy'], default='roc_auc')

parser.add_argument('-r',"--ratioTrainTest", help="Ratio Train/Test used (default : 0, no Test)", type=float, default=0)

parser.add_argument("--cardPatch", help="Number of Patch you want to extract", type=int, default=10000)

parser.add_argument("--compressSize", help="Size of features compression (default : 2)", type = int, default=100)

parser.add_argument("--dimReduce", help="Type of dimension reducing", choices=['fisher','lda','pca','selec'])

parser.add_argument("--operationStr", help="Type of operation to apply to patches",default='mean')
parser.add_argument("--outputFormat", help="type of output for stft matrix file (default : npy)", default="npy", choices=['npy','mat'])

parser.add_argument("-c", "--copyResults", help="Copy Results to your home if argument presnt", action="store_true") 
parser.add_argument("-t", "--transfer", help="Transfer Learning if indicated", action="store_true") 

parser.add_argument("-n", "--normalize", help="Type of normalizing/Scaling",choices=["standard","sample","cov",None])



args = parser.parse_args()
print(args,'\n')

data = args.data.lower()
model = args.model

if args.subject in ('Y','Z'):
    CARDELEC = 32
else:
    CARDELEC = 64
#=========================  DATA  ==================================
#===================================================================
if data == 'test':

    dataType = 'raw'
    X = np.array([[i for i in range(64*4)] for j in range(50)])
    X = np.concatenate((X, -X))

    y = np.array([1 for j in range(50)])
    y = np.concatenate((y,-y))

    xTest = X[10:,:]
    yTest = y[10:]

elif data == 'random':
    y = np.load(PATH_TO_DATA+'AfullY.npy')
    X = np.random.random((np.size(y),args.cardRandom))

    xTest = X[:10]
    yTest = y[:10]
    
    dataType = 'random'

elif data == 'r':
    
    X,y,xTest,yTest = prepareRaw(args.subject,args.ratioTrainTest)
    dataType='raw'

elif data == 's':

    X,y,xTest,yTest = prepareStft(args.subject,args.sizeWindow,args.ratioTrainTest)
    dataType="stft{}".format(args.sizeWindow)

elif data=='f':

    subject=args.subject
    freqMin=args.freqMin
    freqMax = args.freqMax
    decimation = args.decimation
    
    X,y,xTest,yTest = prepareFiltered(subject,freqMin,freqMax,decimation,args.ratioTrainTest)
    dataType = '{}filtered{}{}{}'.format(subject,freqMin,freqMax,decimation)

elif data == 'fs':

    subject=args.subject
    freqMin=args.freqMin
    freqMax = args.freqMax
    decimation = args.decimation
    sizeWindow = args.sizeWindow
    
    X,y,xTest,yTest = prepareFilteredStft(subject, freqMin, freqMax, decimation,args.sizeWindow, args.ratioTrainTest)
    dataType = '{}filtered{}{}{}Stft{}'.format(subject,freqMin,freqMax,decimation,sizeWindow)

elif data == 'p':

    cardPatch = args.cardPatch

    X,y,xTest,yTest =  preparePatch(args.subject, args.freqMin, args.freqMax, args.decimation,args.sizeWindow,cardPatch, args.ratioTrainTest,args.outputFormat,args.operationStr)
    dataType = 'patched{}'.format(cardPatch)

else :
    print(USAGE)
    raise ArgumentError("Wrong Type of Data")

if args.transfer:

    X,y,xTest,yTest = prepareTransfertFiltered(0.5,30,4)
    dataType = 'filtered4'

    print("+Transfer")

dataType += "Balanced"
print(dataType)

#================= Normalisation des données =====================
#=================================================================
if args.dimReduce:

    if np.size(xTest)==0:
        raise NoTestError("A test file is needed for dimension reducing, otherwise, test would be biaised.\nAdd '-r 0.8' option when calling mainParams.py")

    compressSize = args.compressSize

    if args.dimReduce == 'fisher':
        X,xTest = fisherCriterionFeaturesSelection(X,y,xTest,compressSize)
        dataType += "fisher{}".format(compressSize)

    elif args.dimReduce == 'lda':
        X,xTest = transformLDA(X,y,xTest)
        dataType += "lda{}".format(compressSize)

    if args.dimReduce == 'pca':
        X,xTest = dimensionReducePCA(X,xTest,compressSize)
        dataType += "lda{}".format(compressSize)

    if args.dimReduce == 'selec':
        X,xTest = selectElectrode(X,xTest)
        

else:
    X,xTest = normalizeData(X,xTest,CARDELEC,args.normalize)        
#====================================================================
#=========================  MODEL  ==================================
#====================================================================
#====================================================================
cardJobs=args.jobs
scoring=args.scoring

if model=='lin':
    
    learnHyperLinear(X, y, xTest, yTest, 'l2', scoring,transformedData=dataType,jobs=cardJobs)

elif model=='nonLin':
    
    learnHyperNonLinear(X, y, xTest, yTest, scoring,transformedData=dataType,jobs=cardJobs)

elif model == 'elastic' :

    learnElasticNet(X,y,xTest,yTest, scoring, transformedData=dataType, jobs=cardJobs)
    
elif model == 'forest' :

    learnRandomForest(X,y,xTest,yTest,scoring,jobs=cardJobs)
    
elif model=='csp':

    learnLinCspBiaised(X,y,xTest,yTest,scoring,transformedData=dataType,jobs=cardJobs)

elif model == 'pipeLin':

    learnCspPipeline(X,y,xTest,yTest,scoring,transformedData=dataType,jobs=cardJobs,classifier='lin')

elif model == 'pipeRidge':

    learnCspPipeline(X,y,xTest,yTest,scoring,transformedData=dataType,jobs=cardJobs,classifier='ridge')


elif model == 'ridge':
    learnRidge(X,y,xTest,yTest,scoring,dataType,cardJobs)

elif model == 'step':
    learnStep(X, y, xTest, yTest, 'l2', scoring, transformedData=dataType,jobs=cardJobs)

elif model == 'elec':
    learnElecFaster(X, y, xTest, yTest, 'l2', scoring, transformedData=dataType,jobs=cardJobs)

else :
    raise ArgumentError("Wrong Type of Model")

if args.copyResults:
    os.system("cp Results/*.txt ~")

