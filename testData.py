#!/usr/bin/env python
# -*- coding: utf-8 -*

PATHTODATA = 'BCI/'
PATHTOMODEL = 'Models/'

from rawManipulation import *
from sklearn import svm
import pickle

def savePartOfData(filenameI, filenameO):
    data = sio.loadmat("{}{}".format(PATHTODATA, filenameI))
    miniData = np.array(data['X'][0])
    miniData = miniData.transpose()
    y = [elem[0] for elem in data['y']]

    np.save(filenameO, miniData)
    np.save('{}fullY'.format(PATHTODATA), y)

def reformatStftData(filenameI):
    #TODO
    data = sio.loadmat(filenameI)
    data = np.array(data['X'])

def reformatRawData(filenameI, filenameO):
    data = sio.loadmat("{}{}".format(PATHTODATA, filenameI))
    data = np.array(data['X'])

    numExemple = np.size(data,2)
    newData = np.empty([numExemple, 64*160])

    for exemple in range(np.size(data,2)):
        newData[exemple, :] = np.concatenate( [data[i,:,exemple] for i in range(64)])

    np.save('{}{}'.format(PATHTODATA, filenameO), newData)

def saveSplitted(trainX, trainY, testX, testY):
    np.save('{}trainRawX'.format(PATHTODATA), trainX)
    np.save('{}trainY'.format(PATHTODATA), trainY)
    np.save('{}cvRawX'.format(PATHTODATA), testX)
    np.save('{}cvY'.format(PATHTODATA), testY)


def splitXY(fileX='AfullRawX.npy', fileY='AfullY.npy', split=0.6, random=False):
    X = np.load('{}{}'.format(PATHTODATA, fileX))
    y = np.load('{}{}'.format(PATHTODATA, fileY))

    numIndex = int(np.size(X,0)*split)

    if not random:
        saveSplitted(X[:numIndex], y[:numIndex], X[numIndex:], y[numIndex:])
    else:
        indexes = np.random.random_integers(0,np.size(X,0)-1, numIndex)
        saveSplitted(X[indexes], y[indexes], np.delete(X,indexes,0), np.delete(y,indexes,0))

def learnData(fileX='trainRawX.npy', fileY='trainY.npy', C=1, modelType='linear'):
    X = np.load("{}{}".format(PATHTODATA, fileX))
    y = np.load("{}{}".format(PATHTODATA, fileY))

    if modelType == 'linear':
        clf = svm.LinearSVC(C=C)
    else:
        clf = svm.SVC(C=C)

    print("SVM begin.")
    clf.fit(X,y)
    pickle.dump(clf, open('{}model{}{}'.format(PATHTOMODEL, C, modelType), 'wb'))


def evalModel(fileTrainX='trainRawX.npy', fileTrainY='trainY.npy',fileCvX='cvRawX.npy', fileCvY='cvY.npy', C=1, modelType='linear'):
    def _subEval(X, y, classif, C, modelType, dataset):

        numExemple = np.size(X,0)
        predX = clf.predict(X)
        results = predX == y
        score = np.size(np.where(results))/numExemple
        print("for model {} C: {} => {:.4} % of success on {} Dataset".format(modelType, C, score*100, dataset))
        return score


    clf = pickle.load( open('{}model{}{}'.format(PATHTOMODEL, C, modelType), 'rb') )
    print("Model Loaded")

    Xtrain = np.load('{}{}'.format(PATHTODATA, fileTrainX))
    ytrain = np.load('{}{}'.format(PATHTODATA, fileTrainY))

    Xcv = np.load('{}{}'.format(PATHTODATA, fileCvX))
    ycv = np.load('{}{}'.format(PATHTODATA, fileCvY))

    _subEval(Xtrain, ytrain, clf, C, modelType, 'train')
    _subEval(Xcv, ycv, clf, C, modelType, 'Cross Val')


def fullTest(C=1, modelType='linear'):
    learnData(C=C, modelType=modelType)
    evalModel(C=C, modelType=modelType)


def testTemplate(models, numRepetition, resplit=True, linear=True, nonlinear=False):

    for c in models:
        for repet in range(numRepetition):
            if resplit:
                print('Splitting Data')
                splitXY(split=0.7, random=True)
                print('Data Splitted')
            if linear:
                fullTest(C=c, modelType='linear')
            if nonlinear:
                fullTest(C=c, modelType='nonlinear')


models = [2, 5]
testTemplate(models, 2, resplit=True, linear=False, nonlinear=True)
