#!/usr/bin/env python
# -*- coding: utf-8 -*
#Perso
from signalManipulation import *
from manipulateData import *

#Module
import pickle

from sklearn import svm, grid_search
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.cross_validation import StratifiedKFold

from copy import copy,deepcopy

import pylab as pl

#======================== TOOLS ========================
#======================================================
def writeResults(results, best_params, best_score, modelType, penalty, scoreType,\
                 transformedData, scores=None):
    """
    Write results of a grid_search in a file
    [parameters] [score] [STD]
    ...
    [Confusion Matrix of the best model on train]
    [Confusion Matrix of the best model on test]
    Best Params : XXXX Score CV : XXX%
    Accuracy Train : XX Accuracy Test : XX
    F1 Train : XX F1 Test : XX

    Ex :

    1.3 0.91
    1.7 0.65
    [[9787    4]
     [ 399  520]]
    [[6690  276]
     [ 598  30]]
    Best Params : 1.3 Score CV : 0.91
    Accuracy Train : 0.91  Accuracy Test : 0.80
    F1 Train : 0.80 F1 Test : 0.50
    """

    strScores = ""

    if modelType=='NonLinear':
        for model in results:
            print(model)
            strScores += "{:.4} {} {} {}\n".format(model[0]['C'], model[0]['gamma'], model[1], np.std(model[2]))
    elif modelType=='ElasticNet':
        for model in results:
            print(model)
            strScores += "{:.4} {} {} {}\n".format(model[0]['alpha'], model[0]['l1_ratio'], model[1], np.std(model[2]))

    elif modelType=='Pipe':
        for model in results:
            print(model)
            if 'classif__C' in model[0].keys():
                strScores += "{} {:.4} {} {}\n".format(model[0]['csp__n_components'], model[0]['classif__C'], model[1], np.std(model[2]))
            else:
                strScores += "{} {:.4} {} {}\n".format(model[0]['csp__n_components'], model[0]['classif__alpha'], model[1], np.std(model[2]))

    elif modelType=='Ridge':
        for model in results:
            print(model)
            strScores += "{:.4} {} {}\n".format(model[0]['alpha'], model[1], np.std(model[2]))

                
    else: #Linear, C is the only parameter
        for model in results:
            print(model)
            strScores += "{:.4} {} {}\n".format(model[0]['C'], model[1], np.std(model[2]))
 

    strScores += "Best Params : {} Score CrossVal : {} \n".format(best_params, best_score)

    if scores:
        strScores += "{}\n{}\n".format(str(scores['cMatrixTrain']),\
                                       str(scores['cMatrixTest']))

        strScores += "Accuracy Train : {}  Accuracy Test : {} \n".format(scores['accTrain'], scores['accTest'])
        strScores += "F1 Train : {}  F1 Test : {} \n".format(scores['f1Train'],\
                                                             scores['f1Test'])
        strScores += "Roc_Auc Train : {}  Roc_Auc Test : {} \n".format(scores['rocTrain'],scores['rocTest'])
    else:
        print("No Test file")
        strScores += "\nNo Test file\n=========\n"
        
    f = open("{}{}HyperSelection{}{}{}.txt".format(RESULTS_PATH, penalty, modelType.title(), scoreType.title(), transformedData.title()), 'w')
    f.write(strScores)
    f.close()

def getScores(y, yPredTrain, yTest, yPredTest):

    scores = dict()

    scores['f1Train'] = f1_score(y, yPredTrain)
    scores['f1Test'] = f1_score(yTest, yPredTest)


    scores['accTrain'] = accuracy_score(y, yPredTrain)
    scores['accTest'] = accuracy_score(yTest, yPredTest)
    

    scores['rocTrain'] = roc_auc_score(y, yPredTrain)
    scores['rocTest'] = roc_auc_score(yTest, yPredTest)
    

    scores['cMatrixTrain'] = confusion_matrix(y, yPredTrain)
    scores['cMatrixTest'] = confusion_matrix(yTest, yPredTest)

    proba = float(len(np.where(y==1)[0]))/len(y)
    if proba < 0.50:
        proba = 1 - proba
    scores['random'] = proba
    
    return scores

def printScores(scores):

    strSave =  "Train :\n"
    strSave += "Accuracy : {}\n".format(scores['accTrain'])
    strSave += "Roc_Auc :  {}\n".format(scores['rocTrain'])
    strSave += "F1 :       {}\n".format(scores['f1Train'])
    strSave += "{}\n".format(scores['cMatrixTrain'])

    strSave += "Test :\n"
    strSave += "Accuracy : {}\n".format(scores['accTest'])
    strSave += "Roc_Auc :  {}\n".format(scores['rocTest'])
    strSave += "F1 :       {}\n".format(scores['f1Test'])
    strSave += "{}\n".format(scores['cMatrixTest'])

    strSave += "Random Accuracy : {}".format(scores['random'])

    
    print strSave
    return strSave


def testModel(best,X,y,xTest,yTest,penalty):
    
    print("Predicting Data :")
    yPredTrain = best.predict(X)
    yPredTest = best.predict(xTest)
    scores = getScores(y, yPredTrain, yTest, yPredTest)
    printScores(scores)

    if penalty=='l1':
        saveNonZerosCoef(best, 'l1', dataType=transformedData)
        analyzeCoef(dataType=transformedData, reg='l1')

    return scores


def saveNonZerosCoef(clf, reg, dataType):

    nonZerosParams = np.where(clf.coef_ != 0)[0]
    print("Nombre de coef : ", len(clf.coef_[0]))
    print("Nombre de coef annulés : ", len(nonZerosParams))

    with open('nonZerosParams{}{}'.format(dataType.title(),reg), 'w') as f:
        f.write(str(list(nonZerosParams)))

    analyzeCoef(dataType, reg)


def analyzeCoef(dataType, reg):

    path = "Images/Screenshots/"
    
    with open('nonZerosParams{}{}'.format(dataType.title(),reg), 'r') as f:
        wholeFile = f.read()
        print("Here")
        print(wholeFile[0], wholeFile[-1])
        wholeFile = wholeFile[1:-1]
        numGen = map(int,wholeFile.split(','))

        #Step
        step = np.zeros(40)
        steps = np.array([i+1 for i in range(40)])
        for num in numGen:
            step[num%40] += 1

        numGen = map(int,wholeFile.split(','))

        #Elec
        elec = np.zeros(64)
        elecs = np.array([i+1 for i in range(64)])

        for num in numGen:
            elec[num//40] += 1

        ax = plt.subplot()

        steps = np.array(steps)/60
        
        ax.bar(steps, step, width=1/60)
        ax.set_title("Nombre de coefficients non annulés par pas de temps")
        plt.savefig(path+'nonZerosStep{}{}.png'.format(dataType.title(),reg))

        plt.show()
        
        ax = plt.subplot()
        ax.bar(elecs, elec, width=1)
        ax.set_title("Nombre de coefficients non annulés par electrode")
        plt.savefig(path+'nonZerosElec{}{}.png'.format(dataType.title(),reg))
        plt.show()

#=============== Learner =============================
#====================================================
def learnHyperLinear(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):
    """
    Grid Search over a set of parameters for linear model
    """
    #Check if test is empty, if it is, don't refit and predict data
    testAvailable = np.size(xTest,0)!=0

    # Parameters selection
    #====================
    cRange = np.logspace(-5,1,3)
    parameters = {'C': cRange}

    if penalty=='l1':
        dual=False
    else:
        dual=True

    #Creating Model and begin classification
    #=======================================
    classif = svm.LinearSVC(penalty=penalty, class_weight=CLASS_WEIGHT,  dual=dual)
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=jobs, verbose=3, refit=testAvailable)
    print("Begin\n...")
    clf.fit(X,y)

    
    #Get results, print and write them into a file
    #============================================
    print(clf.best_params_, clf.best_score_)

    if testAvailable:
        scores = testModel(clf.best_estimator_,X,y,xTest,yTest,penalty)
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,'Linear',\
                     penalty,scoring, transformedData, scores=scores)
    else:
        print("No test, don't predict data")
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,'Linear',\
                     penalty,scoring, transformedData, scores=None)
    


def learnHyperNonLinear(X, y, xTest, yTest, scoring, transformedData,jobs=1):
    """
    Grid Search over a set of parameters for a non-linear model
    """
    #Check if test is empty, if it is, don't refit and predict data
    testAvailable = np.size(xTest,0)!=0
    

    # Parameters selection
    #====================
    cRange = np.logspace(-5,2,8)
    gRange = np.logspace(-5,2,8)
    parameters = {'C': cRange, 'gamma':gRange}
    
    #Creating Model and begin classification
    #=======================================
    classif = svm.SVC(class_weight=CLASS_WEIGHT)
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=jobs,verbose=3,refit=testAvailable)
    print("Begin\n...")
    clf.fit(X,y)

    #Get results, print and write them into a file
    #============================================
    print(clf.best_params_, clf.best_score_)
        
    if testAvailable:
        scores = testModel(clf.best_estimator_,X,y,xTest,yTest,'l2')
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
                     'NonLinear', 'l2', scoring, transformedData, scores=scores)

        
    else:
        print("No test, don't predict data")
    
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
        'NonLinear', 'l2', scoring, transformedData, scores=None)

def learnRidge(X,y,xTest,yTest,scoring, transformedData, jobs):
    """
    Grid Search over a set of parameters for linear model
    """
    #Check if test is empty, if it is, don't refit and predict data
    testAvailable = np.size(xTest,0)!=0

    # Parameters selection
    #====================
    alpha = np.logspace(-3,3,6)
    parameters = {'alpha': alpha}

    #Creating Model and begin classification
    #=======================================
    classif = RidgeClassifier(class_weight=CLASS_WEIGHT)
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=10, n_jobs=jobs, verbose=3, refit=testAvailable)
    print("Begin\n...")
    clf.fit(X,y)

    #Get results, print and write them into a file
    #============================================
    print(clf.best_params_, clf.best_score_)

    if testAvailable:
        scores = testModel(clf.best_estimator_,X,y,xTest,yTest,'l2')
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,'Ridge',\
                     'l2',scoring, transformedData, scores=scores)
    else:
        print("No test, don't predict data")
        writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,'Ridge',\
                     'l2',scoring, transformedData, scores=None)
    

def learnRandomForest(X,y,xTest,yTest,scoring, jobs):

    params = {
        'n_estimators':[2,10,100],
        'max_features':['auto',2,10],
        'max_depth':[10,40,2],
        'min_samples_split':[2,10,20,50]
    }
    
    forest = RandomForestClassifier()

    grd = grid_search.GridSearchCV(forest,params, scoring=scoring,cv=3,n_jobs=jobs,verbose=3)
    grd.fit(X,y)

    yPredTrain = grd.predict(X)
    yPredTest = grd.predict(xTest)

    print "FOREST : \n"
    scores = getScores(y, yPredTrain, yTest, yPredTest)
    printScores(scores)


def learnCspPipeline(X, y, xTest, yTest, scoring,transformedData,jobs=1, classifier='lin'):

    testAvailable = np.size(xTest)
    
    X = vecToMat(X)

    if testAvailable:
        xTest = vecToMat(xTest)

    if classifier=='lin':
        classif = svm.LinearSVC(penalty='l2',class_weight=CLASS_WEIGHT)
        params = np.logspace(-5,1,3)
        hyper = 'classif__C'

    else:
        classif = RidgeClassifier(class_weight=CLASS_WEIGHT)
        params = np.logspace(-1,3,10)
        hyper = 'classif__alpha'

    csp = CSP(reg='ledoit_wolf',log=False)
    scaler = StandardScaler()
    pipe = Pipeline(steps = [('csp',csp), ('scaler',scaler), ('classif',classif)])
    pipe = Pipeline(steps = [('csp',csp), ('classif',classif)])

    n_components = [1,2,5,10,20,30,40,50]
    dico = {'csp__n_components':n_components, hyper:params}

    grd = grid_search.GridSearchCV(pipe,dico, cv=5, verbose=3, n_jobs=4)
    grd.fit(X,y)

    
    if testAvailable:
        scores = testModel(grd.best_estimator_,X,y,xTest,yTest,'l2')
        writeResults(grd.grid_scores_, grd.best_params_, grd.best_score_,'Pipe', 'l2', scoring, transformedData, scores=scores)

    else:
        print("No test, don't predict data")    
        writeResults(grd.grid_scores_, grd.best_params_, grd.best_score_,'Pipe', 'l2', scoring, transformedData, scores=None)


                
def learnElasticNet(X,y,xTest,yTest,scoring,transformedData='raw',jobs=1):

    # Parameters selection
    #====================
    alpha = np.linspace(0.01,0.2,5)
    l1_ratio = np.linspace(0.01,0.3,5)
    parameters = {'alpha': alpha, 'l1_ratio': l1_ratio}
    
    #Creating Model and begin classification
    #=======================================
    classif = ElasticNet(selection='random')
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=jobs,verbose=3)

    print("Begin\n...")
    clf.fit(X,y)

    #Get results, print and write them into a file
    #============================================
    best = clf.best_estimator_
    print(clf.best_params_, clf.best_score_)

    if np.size(a,0)!=0:
        print("Predicting Data :")
        yPredTrain = best.predict(X)
        yPredTrain[yPredTrain >= 0] = 1
        yPredTrain[yPredTrain < 0] = -1

        yPredTest = best.predict(xTest)
        yPredTest[yPredTest >= 0] = 1
        yPredTest[yPredTest < 0] = -1

        scores = getScores(y, yPredTrain, yTest, yPredTest)
        printScores(scores)
    
    writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
                 'ElasticNet', 'l1l2', scoring, transformedData, scores)
    
    nonZerosParams = np.where(best.coef_ != 0)[0]
    print(len(nonZerosParams))
    print(nonZerosParams)

    with open('nonZerosParamsRawElasticNet', 'w') as f:
        f.write(str(list(nonZerosParams)))

def learnStep(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):

    baseClf = svm.LinearSVC(penalty='l2', class_weight=CLASS_WEIGHT)
    cRange = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1, 10]
    parameters = {'C': cRange}

    best_score = 0
    numStep = np.size(X,1)//64
    keptStep = np.ones(numStep, dtype=bool)
    copyX = copy(X)
    copyXTest = copy(xTest)

    scores = np.zeros(numStep)
    scoreDecrease = False
    numFailed = 0
    
    while not scoreDecrease:

        scores[:] = 0

        for step in range(numStep):
            if not keptStep[step] :
                continue
            else:
                erased = list(np.where(keptStep==False)[0])
                
                if erased != []:
                    erased.append(step)
                    X = delTimeStep(X, erased, transformedData)
                    xTest = delTimeStep(xTest, erased, transformedData)
                else:
                    X = delTimeStep(X,step, transformedData)
                    xTest = delTimeStep(xTest, step, transformedData)

                print("Learning Model without step N°",step)

                clf = grid_search.GridSearchCV(baseClf, parameters, scoring=scoring,\
                                               cv=5, n_jobs=jobs, verbose=3)
                clf.fit(X,y)

                best = clf.best_estimator_
                print(clf.best_params_, clf.best_score_)

                yPredTest = best.predict(xTest)


                if scoring=='f1':
                    scores[step] = f1_score(yTest, yPredTest)
                else:
                    scores[step] = roc_auc_score(yTest, yPredTest)


                print("Score :", scores[step])

                #post process :
                X = copy(copyX)
                xTest = copy(copyXTest)
                
        worstStep = np.argmax(scores)
        keptStep[worstStep] = False

        print("Score max : {}, removing step N°{}".format(scores[worstStep], worstStep))
        print("Step removed : ", np.where(keptStep==False))
        print("Past Best : ", best_score)

        if scores[worstStep] > best_score:
            best_score = scores[worstStep]
        else:
            numFailed += 1
            
        if numFailed > 3:
            scoreDecrease = True

def learnElecFaster(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):
    
    baseClf = svm.LinearSVC(penalty='l2', class_weight=CLASS_WEIGHT)
    cRange = np.logspace(-5,2,8)
    
    parameters = {'C': cRange}

    if np.size(xTest)!=0:
        X = np.concatenate((X,xTest))
        y = np.concatenate((y,yTest))
        
    # clf = grid_search.GridSearchCV(baseClf, parameters, scoring=scoring, cv=5, n_jobs=jobs, verbose=3)
    # clf.fit(X,y)
    # bestParams = clf.best_params_
    # print(bestParams['C'], clf.best_score_)

    # C = bestParams['C']
    C = 1e-5
    baseClf = svm.LinearSVC(penalty='l2', class_weight=CLASS_WEIGHT)

    best_score = 0
    best_selection = []
    keptElec = np.ones(64, dtype=bool)

    copyX = copy(X)
    
    scores = np.zeros(64)
    scoreDecrease = False
    numFailed = 0
    
    for numIter in range(63):

        scores[:] = 0

        for elec in range(64):
            if not keptElec[elec] :
                #Already deleted
                continue
            else:

                print("Deleting Electrode(s) ...")
                erased = list(np.where(keptElec==False)[0])                
                if erased != []:
                    erased.append(elec)
                    X = delElec(X, erased, transformedData)
                else:
                    X = delElec(X,elec, transformedData)

                print("Learning Model without elec N°",elec)

                clf = grid_search.GridSearchCV(baseClf, {'C':[C]}, scoring=scoring, cv=10, n_jobs=jobs, verbose=1)
                clf.fit(X,y)
                
                scores[elec] = clf.best_score_

                print(scores[elec])
                    
                #post process :
                X = copy(copyX)
                
        worstElec = np.argmax(scores)
        keptElec[worstElec] = False
        removedElec = np.where(keptElec==False)
        print("Score max : {}, removing elec N°{}".format(scores[worstElec], worstElec))
        print("Elec removed : ", removedElec)
        
        print("Past Best : ", best_score, "with : ", best_selection)

        if scores[worstElec] > best_score:
            best_score = scores[worstElec]
            best_selection = np.where(keptElec==False)

        else:
            numFailed += 1

        with open("selecStep.txt",'a') as f:
            f.write("{} : {} with elec {}, numFailed : {}\n".format(numIter, scores[worstElec], removedElec, numFailed))





