#!/usr/bin/env python
# -*- coding: utf-8 -*

#Perso
from signalManipulation import *
from manipulateData import *

#Module
import pickle
from sklearn import svm, grid_search
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

#Const
PATHTODATA = 'BCI/'
PATHTOMODEL = 'Models/'

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

# party = np.array([i for i in range(30)])
# X = np.concatenate((np.array([[-i,-i] for i in range(30)]), -np.array([[-i,-i] for i in range(30)]),\
#                     np.array([[-i,-i] for i in range(40,80)]), -np.array([[-i,-i] for i in range(40,80)])))
# y = np.concatenate((np.array([1 for i in range(30)]), -np.array([1 for i in range(30)]),\
#                     np.array([1 for i in range(40,80)]), -np.array([1 for i in range(40,80)])))

X = np.load('{}{}'.format(PATHTODATA, 'trainRawX.npy'))
y = np.load('{}{}'.format(PATHTODATA, 'trainY.npy'))

print(X.shape)

Xtest = np.load('{}{}'.format(PATHTODATA, 'cvRawX.npy'))
ytest = np.load('{}{}'.format(PATHTODATA, 'cvY.npy'))

gamma_range = np.logspace(-9, 2, 5)
parameters = {'gamma': gamma_range, 'C':[0.8,1,2,3, 1e6, 1e8]}
classif = svm.linearSVC()
clf = grid_search.GridSearchCV(classif, parameters, cv=10, n_jobs=2)
print("Begin\n...")
clf.fit(X,y)
best = clf.best_estimator_
print(clf.best_params_, clf.best_score_)
print("Saving Model")

pickle.dump(best, open('{}modelGridSearch'.format(PATHTOMODEL), 'wb'))

print(r2_score(y, best.predict(X)))
print(r2_score(ytest, best.predict(Xtest)))
