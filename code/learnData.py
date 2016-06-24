#!/usr/bin/env python
# -*- coding: utf-8 -*

#Perso
from signalManipulation import *
from manipulateData import *
#Also import CONSTANT from manipulateData : PATH_TO_DATA etc ...

#Module
import pickle
from sklearn import svm, grid_search
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import scale

from copy import copy

import pylab as pl

RESULTS_PATH = 'Results/'


def writeResults(results, best_params, best_score, modelType, penalty, scoreType,\
                 transformedData, scores):
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
    else: #Linear, C is the only parameters
        for model in results:
            print(model)
            strScores += "{:.4} {} {}\n".format(model[0]['C'], model[1], np.std(model[2]))
 
    strScores += "{}\n{}\n".format(str(scores['cMatrixTrain']), str(scores['cMatrixTest']))

    strScores += "Best Params : {} Score CrossVal : {} \n".format(best_params, best_score)
    strScores += "Accuracy Train : {}  Accuracy Test : {} \n".format(scores['accTrain'],\
                                                                     scores['accTest'])
    strScores += "F1 Train : {}  F1 Test : {} \n".format(scores['f1Train'], scores['f1Test'])
    strScores += "Roc_Auc Train : {}  Roc_Auc Test : {} \n".format(scores['rocTrain'],\
                                                                   scores['rocTest'])

    
    f = open("{}{}HyperSelection{}{}{}.txt".format(RESULTS_PATH, penalty, modelType.title(), scoreType.title(), transformedData.title()), 'w')
    f.write(strScores)
    f.close()

def getScores(y, yPredTrain, yTest, yPredTest):

    scores = dict()

    scores['f1Train'] = f1_score(y, yPredTrain)
    scores['f1Test'] = f1_score(yTest, yPredTest)

    print("F1 :\n Train = {} Test : = {}".format(scores['f1Train'], scores['f1Test']))

    scores['accTrain'] = accuracy_score(y, yPredTrain)
    scores['accTest'] = accuracy_score(yTest, yPredTest)
    
    print("Accuracy :\n Train = {} Test : = {}".format(scores['accTrain'], scores['accTest']))

    scores['rocTrain'] = roc_auc_score(y, yPredTrain)
    scores['rocTest'] = roc_auc_score(yTest, yPredTest)
    
    print("Roc_Auc :\n Train = {} Test : = {}".format(scores['rocTrain'], scores['rocTest']))

    scores['cMatrixTrain'] = confusion_matrix(y, yPredTrain)
    scores['cMatrixTest'] = confusion_matrix(yTest, yPredTest)

    print(scores['cMatrixTrain'],'\n',scores['cMatrixTest'])

    return scores

def learnHyperLinear(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):
    """
    Grid Search over a set of parameters for linear model
    """
    # Parameters selection
    #====================
    cRange = np.logspace(-5,1,7)
    parameters = {'C': cRange}

    if penalty=='l1':
        dual=False
    else:
        dual=True

    #Creating Model and begin classification
    #=======================================
    classif = svm.LinearSVC(penalty=penalty, class_weight='balanced',  dual=dual)
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=jobs, verbose=3)
    print("Begin\n...")
    clf.fit(X,y)

    
    #Get results, print and write them into a file
    #============================================
    best = clf.best_estimator_
    print(clf.best_params_, clf.best_score_)

    print("Predicting Data :")
    yPredTrain = best.predict(X)
    yPredTest = best.predict(xTest)

    scores = getScores(y, yPredTrain, yTest, yPredTest)
    
    writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
                 'Linear', penalty, scoring, transformedData, scores)

    if penalty=='l1':
        saveNonZerosCoef(best, 'l1', dataType=transformedData)
        analyzeCoef(dataType=transformedData, reg='l1')


def learnHyperNonLinear(X, y, xTest, yTest, scoring, transformedData,jobs=1):
    """
    Grid Search over a set of parameters for a non-linear model
    """
    # Parameters selection
    #====================
    cRange = np.logspace(-5,1,7)
    gRange = np.logspace(-5,1,7)
    parameters = {'C': cRange, 'gamma':gRange}
    
    #Creating Model and begin classification
    #=======================================
    classif = svm.SVC(class_weight='balanced')
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=jobs,verbose=3)
    print("Begin\n...")
    clf.fit(X,y)

    #Get results, print and write them into a file
    #============================================
    best = clf.best_estimator_
    print(clf.best_params_, clf.best_score_)

    print("Predicting Data :")
    yPredTrain = best.predict(X)
    yPredTest = best.predict(xTest)

    scores = getScores(y, yPredTrain, yTest, yPredTest)
    
    writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
                 'NonLinear', 'l2', scoring, transformedData, scores)

                
def learnElasticNet(X,y,xTest,yTest,scoring,transformedData='raw',jobs=1):

    # Parameters selection
    #====================
    alpha = np.linspace(0.01,0.2,5)
    l1_ratio = np.linspace(0.01,0.3,5)
    parameters = {'alpha': alpha, 'l1_ratio': l1_ratio}
    
    #Creating Model and begin classification
    #=======================================
    classif = ElasticNet(selection='random')
    clf = grid_search.GridSearchCV(classif, parameters, scoring=scoring, cv=5, n_jobs=5,verbose=3)

    print("Begin\n...")
    clf.fit(X,y)

    #Get results, print and write them into a file
    #============================================
    best = clf.best_estimator_
    print(clf.best_params_, clf.best_score_)

    print("Predicting Data :")
    yPredTrain = best.predict(X)
    yPredTrain[yPredTrain >= 0] = 1
    yPredTrain[yPredTrain < 0] = -1
    
    yPredTest = best.predict(xTest)
    yPredTest[yPredTest >= 0] = 1
    yPredTest[yPredTest < 0] = -1
    
    scores = getScores(y, yPredTrain, yTest, yPredTest)
    
    writeResults(clf.grid_scores_, clf.best_params_, clf.best_score_,\
                 'ElasticNet', 'l1l2', scoring, transformedData, scores)
    
    nonZerosParams = np.where(best.coef_ != 0)[0]
    print(len(nonZerosParams))
    print(nonZerosParams)

    with open('nonZerosParamsRawElasticNet', 'w') as f:
        f.write(str(list(nonZerosParams)))

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

def learnStep(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):

    baseClf = svm.LinearSVC(penalty='l2', class_weight='balanced')
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

def learnElec(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):

    baseClf = svm.LinearSVC(penalty='l2', class_weight='balanced')
    cRange = [1e-6, 1e-5, 1e-4, 1e-3]
    parameters = {'C': cRange}

    best_score = 0
    best_selection = []
    keptElec = np.ones(64, dtype=bool)

    copyX = copy(X)
    copyXTest = copy(xTest)

    scores = np.zeros(64)
    scoreDecrease = False
    numFailed = 0
    
    while not scoreDecrease:

        scores[:] = 0

        for elec in range(64):
            if not keptElec[elec] :
                continue
            else:
                erased = list(np.where(keptElec==False)[0])
                
                if erased != []:
                    erased.append(elec)
                    X = delElec(X, erased, transformedData)
                    xTest = delElec(xTest, erased, transformedData)
                else:
                    X = delElec(X,elec, transformedData)
                    xTest = delElec(xTest, elec, transformedData)

                print("Learning Model without elec N°",elec)

                clf = grid_search.GridSearchCV(baseClf, parameters, scoring=scoring,\
                                               cv=5, n_jobs=jobs, verbose=1)
                clf.fit(X,y)

                best = clf.best_estimator_
                print(clf.best_params_, clf.best_score_)

                yPredTest = best.predict(xTest)

                if scoring=='f1':
                    scores[elec] = f1_score(yTest, yPredTest)
                    print('ROC : ',roc_auc_score(yTest, yPredTest))

                else:
                    scores[elec] = roc_auc_score(yTest, yPredTest)

                print(scores[elec])
                    
                #post process :
                X = copy(copyX)
                xTest = copy(copyXTest)
                
        worstElec = np.argmax(scores)
        keptElec[worstElec] = False

        print("Score max : {}, removing elec N°{}".format(scores[worstElec], worstElec))
        print("Elec removed : ", np.where(keptElec==False))

        print("Past Best : ", best_score, "with : ", best_selection)

        if scores[worstElec] > best_score:
            best_score = scores[worstElec]
            best_selection = np.where(keptElec==False)

        else:
            numFailed += 1


        if numFailed > 15:
            scoreDecrease = True

def splitTrainValidateTest(X,y):
    train = 0.5
    validate = 0.25
    test = 0.25

    assert(train+validate+test==1)

    numExemples = np.size(X,0)

    randSelection = np.random.sample(numExemples) 

    trainIndex = np.where(randSelection <= train)
    selectionIndex = np.where(np.logical_and(train < randSelection, randSelection <= train+validate))
    testIndex = np.where(randSelection > train+validate)

    sumIndex = np.size(trainIndex)+np.size(selectionIndex)+np.size(testIndex)
    
    print("Train : {}, Val : {}, Test : {}, Total = {}".format(np.size(trainIndex), np.size(selectionIndex),np.size(testIndex),sumIndex ))

    assert(sumIndex==numExemples)
    
    return X[trainIndex], y[trainIndex], X[selectionIndex], y[selectionIndex], X[testIndex], y[testIndex]


def learnElecUnbiasedFaster(X, y, xTest, yTest, penalty, scoring, transformedData,jobs=1):
    
    baseClf = svm.LinearSVC(penalty='l2', class_weight='balanced')
    cRange = np.logspace(-5,2,8)
    
    parameters = {'C': cRange}

    X, y, xSelec, ySelec, xTest, yTest = splitTrainValidateTest(np.concatenate((X, xTest)),\
                                                            np.concatenate((y,yTest)))
    
    # clf = grid_search.GridSearchCV(baseClf, parameters, scoring=scoring, cv=5, n_jobs=jobs, verbose=3)
    # clf.fit(X,y)
    # bestParams = clf.best_params_
    # print(bestParams['C'], clf.best_score_)

    # C = bestParams['C']
    C = 1e-5
    baseClf = svm.LinearSVC(C=C, penalty='l2', class_weight='balanced')

    best_score = 0
    best_selection = []
    keptElec = np.ones(64, dtype=bool)

    copyX = copy(X)
    copyXSelec = copy(xSelec)
    copyXTest = copy(xTest)

    scores = np.zeros(64)
    scoreDecrease = False
    numFailed = 0
    
    while not scoreDecrease:

        scores[:] = 0

        for elec in range(64):
            if not keptElec[elec] :
                continue
            else:
                erased = list(np.where(keptElec==False)[0])
                
                if erased != []:
                    erased.append(elec)
                    print("X ",np.size(X,1))
                    print("xSelec", np.size(xSelec,1))
                    X = delElec(X, erased, transformedData)
                    xSelec = delElec(xSelec, erased, transformedData)
                    print("X ",np.size(X,1))
                    print("xSelec", np.size(xSelec,1))
                else:
                    X = delElec(X,elec, transformedData)
                    xSelec = delElec(xSelec, elec, transformedData)

                print("Learning Model without elec N°",elec)

                baseClf.fit(X,y)
                yPredSelec = baseClf.predict(xSelec)

                if scoring=='f1':
                    scores[elec] = f1_score(ySelec, yPredSelec)
                    print('ROC : ',roc_auc_score(ySelec, yPredSelec))

                else:
                    scores[elec] = roc_auc_score(ySelec, yPredSelec)

                print(scores[elec])
                    
                #post process :
                X = copy(copyX)
                xSelec = copy(copyXSelec)
                
        worstElec = np.argmax(scores)
        keptElec[worstElec] = False

        print("Score max : {}, removing elec N°{}".format(scores[worstElec], worstElec))
        print("Elec removed : ", np.where(keptElec==False))

        print("Past Best : ", best_score, "with : ", best_selection)

        ##True Test
        X = delElec(X, erased, transformedData)
        xTest = delElec(xTest, erased, transformedData)
        baseClf.fit(X,y)
        yPredTest = baseClf.predict(xTest)
        print("\nTrue Test :\nF1 : {}\nROC : {}\nXXXXXXXXXXXXXXX\n".format(f1_score(yTest, yPredTest), roc_auc_score(yTest, yPredTest)))

        X = copy(copyX)                
        xTest = copy(copyXTest)
        

        if scores[worstElec] > best_score:
            best_score = scores[worstElec]
            best_selection = np.where(keptElec==False)

        else:
            numFailed += 1


        if numFailed > 15:
            scoreDecrease = True

    
