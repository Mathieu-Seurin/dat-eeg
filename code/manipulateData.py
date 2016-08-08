#!/usr/bin/python3
#coding: utf-8


#Perso
from constants import *
from signalManipulation import *

#Module
import pickle
import numpy as np

#Sklearn
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

#Scipy
import scipy.io as sio
from scipy.stats.mstats import winsorize

import os.path
import time

from mne.decoding import CSP
#============================== RAW ===========================================
#==============================================================================
def reformatRawData(filenameI, filenameO):

    #==== If File Already exist, don't recalculate everything : ====
    if os.path.exists('{}{}fullY.npy'.format(PATH_TO_DATA, filenameO[0]))\
       and os.path.exists('{}{}.npy'.format(PATH_TO_DATA, filenameO)) :
        print("Reformat Raw Data : File already exists, loading ...")
        return 0
    #=============================================================
                         
    print("Reformat Raw Data : File doesn't exist, creating ...")

    data = sio.loadmat("{}{}".format(PATH_TO_DATA, filenameI))
    y = data['y']
    y = y.reshape(np.size(y,1))
    X = data['X']

    np.save('{}{}fullY'.format(PATH_TO_DATA, filenameO[0]), y)
    np.save('{}{}'.format(PATH_TO_DATA, filenameO), X)


def filterRawData(filenameI, freqInf, freqSup, decimation):

    if filenameI[1:5] == 'full':
        subject = filenameI[0]
        print("Subject : ",subject)

        if subject== 'Z':
            cardElec = 32
            fs = 2048
        elif subject in ('1','2','3','4','5'):
            cardElec = 64
            fs = 512
        else:
            cardElec = 64
            fs = 240
    else:
        print(filenameI)
        raise NotImplemented("Can't use this format at the moment")

    #==== If File Already exist, don't recalculate everything : ====
    if os.path.exists(PATH_TO_DATA+subject+"fullFiltered{}_{}_{}X.npy".format(freqInf,freqSup,decimation)):
        print("filter Data : File already exists, loading ...")
        return 0
    #===============================================
    
    print("Filter Data : File doesn't exist, creating ...")

    order = 4

    data = np.load(PATH_TO_DATA+filenameI)
    numExemple = np.size(data,0)
    
    numPoints = np.size(data,1)//cardElec
    print('Number of Points per signal : ',numPoints)

    currentSignal = SignalHandler(data[0,:numPoints], fs)
    
    sizeOfNewSignal = np.size(currentSignal.filterSig(order,freqInf,freqSup,decimation),0)
    print('Size of new signal : ', sizeOfNewSignal)

    newData = np.empty((numExemple, cardElec*sizeOfNewSignal))

    for ex, x in enumerate(data):
        for numSignal, i in enumerate(range(0,len(x), numPoints)):
            currentSignal.mainSignal = x[i:i+numPoints]
            signalFiltered = currentSignal.filterSig(order,freqInf,freqSup,decimation)

            indexBegin = numSignal*sizeOfNewSignal
            indexEnd = (numSignal+1)*sizeOfNewSignal
            newData[ex,indexBegin:indexEnd] = signalFiltered

        if not ex%1000:
            print('Transformed : {}/{}'.format(ex, numExemple))
    np.save(PATH_TO_DATA+subject+"fullFiltered{}_{}_{}X".format(freqInf,freqSup,decimation), newData)

#============================== TIME FREQUENCIES===============================
#==============================================================================

def reformatStftData(filenameI, dataType, fs, frameSize=0.2):
    """
    This function should be called after 'reformatRawData'
    filenameI is a string, refering to data in '.npy' format.

    dataType can be either 'raw' or 'filtered'

    fs is the sampling frequencies : Raw signal for the p300 task is sampled at 240Hz

    Framesize is a float, corresponding to the size of window in second (s)
    Default is 0.2
    """

    if filenameI[1:5] == 'full':
        subject = filenameI[0]
        print("Subject : ",subject)

        if subject== 'Z':
            cardElec = 32
        else :
            cardElec = 64

    else:
        print(filenameI)
        raise NotImplemented("Can't use this format at the moment")

    #==== If File Already exist, don't recalculate everything : ====
    if os.path.exists(PATH_TO_DATA+"{}full{}Stft{}X.npy".format(subject, dataType.title(), frameSize)):
        print("Transform Data into STFT : File already exists, loading ...")
        return 0
    #===============================================
    print("Transform Data into STFT : File doesn't exist, creating ...")

    data = np.load(PATH_TO_DATA+filenameI)
    numExemple = np.size(data,0)

    numPoints = np.size(data,1)//cardElec
    print('Number of Points per signal : ',numPoints)

    #============================================
    numFreqBeforeProcess, numWindows = SignalHandler(data[0][0:numPoints],fs).stft(frameSize).shape
    
    freqBeforeProcess = fftfreq(numFreqBeforeProcess*2, 1.0/fs)

    #Remove all negative frequencies and >60Hz (aka noise)
    indexOfKeptFrequencies = np.where(freqBeforeProcess<=60) and np.where(freqBeforeProcess>=0)
    freqs = freqBeforeProcess[indexOfKeptFrequencies]
    numFreqs = np.size(freqs)
    print("Number of Frequencies", numFreqs)
    print("Freqs for this window : ", freqs)

    sizeOfNewSignal = numFreqs*numWindows

    print("""Size of the new signal representation : 
{} frequencies x {} windows = {}""".format(numFreqs ,numWindows, sizeOfNewSignal))
    
    newData = np.empty((numExemple, cardElec*sizeOfNewSignal))
    for ex, x in enumerate(data):
        #Since data is a concatenation of cardElec electrode, you have to separate them
        for numSignal, i in enumerate(range(0,len(x),numPoints)):
            signalStft = SignalHandler(x[i:i+numPoints],fs).stft(frameSize)
            #Stft signal is a matrix [freqs x time] we need to vectorize it
            signalStft = signalStft[indexOfKeptFrequencies,:].flatten()

            indexBegin = numSignal*sizeOfNewSignal
            indexEnd = (numSignal+1)*sizeOfNewSignal
            newData[ex,indexBegin:indexEnd] = signalStft
        
        if not ex%1000:
            print('STFT Transformed : {}/{}'.format(ex, numExemple))

    print("Shape of data (Number of Exemple, Number of Features) :", newData.shape)
    np.save(PATH_TO_DATA+"{}full{}Stft{}X".format(subject, dataType.title(), frameSize), newData)


def reformatStftDataMatrix(filenameI, dataType, fs, frameSize=0.2,numElec=64, outputFormat='npy', includeY=True):
    """
    Args : filenameI, dataType, fs, frameSize=0.2, outputFormat='npy', numElec=64

    Input
    ------

    filenameI is a string, refering to data in '.npy' format.

    dataType can be either 'raw' or 'filteredXX' with XX the order of decimation

    fs is the sampling frequencies : Raw signal for the p300 task is sampled at 240Hz

    Framesize is a float, corresponding to the size of window in second (s)
    Default is 0.2

    outputFormat can be 'npy' (numpy array) or 'mat' matlab matrix format
    
    numElec : Number of Electrodes used in the input file (default is 64, but maybe we wille use CSP, or deleting electrodes, so the number will change)

    Output
    -----
    
    Create a new array, its shape is : (#Exemples x #electrodes x #frequencies x #windows)

    Details
    --------

    This function should be called after 'reformatRawData'
    """
    
    if filenameI[1:5] == 'full':
        subject = filenameI[0]
        print("Subject : ",subject)
    else:
        print(filenameI)
        raise NotImplemented("Can't use this type of file at the moment")

    fileName = PATH_TO_DATA+"{}full{}StftMatrix{}X".format(subject, dataType.title(), frameSize)

    #==== If File Already exist, don't recalculate everything : ====
    if os.path.exists(fileName+'.npy'):
        print("Transform Data into STFT : File already exists, loading ...")
        newData = np.load(fileName+'.npy')

        if outputFormat=='mat':
            if includeY:
                y=np.load('{}{}fullY.npy'.format(PATH_TO_DATA, subject))
                newData = {'X':newData,'y': y}
            sio.savemat(fileName, newData)
            
        return newData
    #===============================================
    print("Transform Data into STFT, non-Vectorized format : File doesn't exist, creating")

    data = np.load(PATH_TO_DATA+filenameI)
    numExemple = np.size(data,0)

    numPoints = np.size(data,1)//numElec
    print('Number of Points per signal : ',numPoints)

    #============================================
    numFreqBeforeProcess, numWindows = SignalHandler(data[0][0:numPoints],fs).stft(frameSize).shape
    
    freqBeforeProcess = fftfreq(numFreqBeforeProcess*2, 1.0/fs)

    #Remove all negative frequencies and >60Hz (aka noise)
    indexOfKeptFrequencies = np.where(freqBeforeProcess<=60) and np.where(freqBeforeProcess>=0)
    freqs = freqBeforeProcess[indexOfKeptFrequencies]
    numFreqs = np.size(freqs)
    print("Number of Frequencies", numFreqs)
    print("Freqs for this window : ", freqs)

    newData = np.empty((numExemple, numElec, numFreqs, numWindows ))

    print("""Size of one signal representation using Stft Matrix Representation : 
    {} electrodes x {} windows x {} frequencies""".format(numElec,numWindows,numFreqs))

    for ex, x in enumerate(data):
        #Since data is a concatenation of numElec electrode, you have to separate them
        #Electrode iteration :
        for numSignal, i in enumerate(range(0,len(x),numPoints)):
            signalStft = SignalHandler(x[i:i+numPoints],fs).stft(frameSize)
            signalStft = signalStft[indexOfKeptFrequencies,:] # Keep only interesting frequencies

            #Stft signal is a matrix [Frequencies x windows]
            newData[ex,numSignal] = signalStft
            
        if not ex%1000:
            print('STFT Transformed : {}/{}'.format(ex, numExemple))

    print("Shape of data (Number of Exemple, Number of Elec, Number of Frequencies, Number of Windows) :", newData.shape)

    if outputFormat=='npy':
        np.save(fileName, newData)
    elif outputFormat=='mat':
        if includeY:
            y=np.load('{}{}fullY.npy'.format(PATH_TO_DATA, subject))
            newData = {'X':newData,'y': y}
        sio.savemat(fileName, newData)
    return newData


def reformatWaveletData(filenameI):
    raise NotImplemented("Not today")


#============================== Whole Process Tools ============================
#==============================================================================
#==============================================================================

# Function for the 1st database : Load subject, reformat, filter data
# split in Train/Test
# return X  y  xTest  yTest

#==============================================================================
def saveSplitted(trainX, trainY, testX, testY):

    np.save(PATH_TO_DATA+'trainX', trainX)
    np.save(PATH_TO_DATA+'trainY', trainY)
    np.save(PATH_TO_DATA+'testX', testX)
    np.save(PATH_TO_DATA+'testY', testY)


def loadSplitted():
        
    X = np.load('{}{}'.format(PATH_TO_DATA, 'trainX.npy'))
    y = np.load('{}{}'.format(PATH_TO_DATA, 'trainY.npy'))
    xTest = np.load('{}{}'.format(PATH_TO_DATA, 'testX.npy'))
    yTest = np.load('{}{}'.format(PATH_TO_DATA, 'testY.npy'))

    return X,y,xTest,yTest

def splitXY(fileX, fileY,split=0.80):

    X = np.load(PATH_TO_DATA+fileX)
    y = np.load(PATH_TO_DATA+fileY)

    if split==0:
        return X,y,np.array([]),np.array([])

    skf = StratifiedKFold(y,5,True)

    for trainIndex, testIndex in skf:

        numExemples = np.size(X,0)
        print("""Splitting Data : \n Train : 80%   Test : 20%
        Total : {} Train : {:10.0f} Test : {:3.0f}""".format(numExemples, numExemples*split, numExemples*(1-split))) 

        saveSplitted(X[trainIndex], y[trainIndex], X[testIndex], y[testIndex])
        return X[trainIndex], y[trainIndex], X[testIndex], y[testIndex]

def concatAB(fileXA, fileYA, fileXB, fileYB, dataType):

    xA, xB = np.load(PATH_TO_DATA+fileXA), np.load(PATH_TO_DATA+fileXB)
    yA, yB = np.load(PATH_TO_DATA+fileYA), np.load(PATH_TO_DATA+fileYB)

    fullX = np.concatenate((xA, xB))
    fullY = np.concatenate((yA, yB))

    np.save(PATH_TO_DATA+"ABfull{}X".format(dataType.title()), fullX)
    np.save(PATH_TO_DATA+"ABfullY", fullY)
    
def prepareFiltered(subject, freqMin, freqMax, decimation, splitTrainTest=0):

    if subject in ('AB','BA'):
        return prepareFilteredAB(freqMin,freqMax,decimation,splitTrainTest)
    
    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject),"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    return splitXY("{}fullFiltered{}_{}_{}X.npy".format(subject,freqMin,freqMax,decimation), "{}fullY.npy".format(subject),splitTrainTest)

def prepareFilteredAB(freqMin,freqMax,decimation,splitTrainTest):

    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    filterRawData("AfullRawX.npy", freqMin, freqMax, decimation)

    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    filterRawData("BfullRawX.npy", freqMin, freqMax, decimation)

    concatAB("AfullFiltered{}_{}_{}X.npy".format(freqMin,freqMax,decimation),"AfullY.npy",\
             "BfullFiltered{}_{}_{}X.npy".format(freqMin,freqMax,decimation),"BfullY.npy",'Filtered')

    return splitXY("ABfullFilteredX.npy","ABfullY.npy",splitTrainTest)
    
def prepareRaw(subject,splitTrainTest=0):

    if subject in ('AB','BA'):
        return prepareRawAB(splitTrainTest)

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    return splitXY("{}fullRawX.npy".format(subject),"{}fullY.npy".format(subject),splitTrainTest)

def prepareRawAB(splitTrainTest):
        
    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    concatAB("AfullRawX.npy","AfullY.npy", "BfullRawX.npy","BfullY.npy",'raw')

    return splitXY("ABfullRawX.npy","ABfullY.npy",splitTrainTest)

                                                     
def prepareStft(subject, frameSize,splitTrainTest):

    if subject in ('AB','BA'):
        prepareStftAB(frameSize,splitTrainTest)

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    reformatStftData("{}fullRawX.npy".format(subject), 'Raw', 240, frameSize=frameSize) 

    return splitXY("{}fullRawStft{}X.npy".format(subject, frameSize),\
                   "{}fullY.npy".format(subject), splitTrainTest)

def prepareStftAB(frameSize,splitTrainTest):
    raise NotImplemented("Not Yet, maybe will never be useful")

def prepareFilteredStft(subject, freqMin, freqMax, decimation,frameSize, splitTrainTest):

    if subject in ('AB', 'BA'):
        prepareFilteredStftAB(freqMin, freqMax, decimation,frameSize,splitTrainTest)

    elif subject in ('Y','Z'):
        fs = 2048
    else:
        fs = 240

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftData(
        "{}fullFiltered{}_{}_{}X.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}'.format(decimation), fs//decimation, frameSize) 

    return splitXY("{}fullFiltered{}Stft{}X.npy".format(subject, decimation, frameSize),\
                   "{}fullY.npy".format(subject), splitTrainTest)

def prepareFilteredStftAB(freqMin, freqMax, decimation,frameSize, splitTrainTest):
    raise NotImplemented("Not Yet, maybe will never be useful")    

def preparePatch(subject, freqMin, freqMax, decimation,frameSize,cardPatch,splitTrainTest,outputFormat,operationStr):
    
    if subject in ('AB','BA'):
        preparePatchAB(freqMin, freqMax, decimation,frameSize,cardPatch,splitTrainTest,outputFormat)
    elif subject in ('Y','Z'):
        fs = 2048
    else:
        fs = 240


    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftDataMatrix(
        "{}fullFiltered{}_{}_{}X.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}_{}_{}'.format(freqMin,freqMax,decimation), fs//decimation, frameSize=frameSize, outputFormat=outputFormat)

    patchProcess(subject, freqMin, freqMax, decimation,frameSize,cardPatch,splitTrainTest,outputFormat,operationStr)

    return splitXY("{}patched{}{}.npy".format(subject,operationStr.title(),cardPatch),"{}fullY.npy".format(subject),splitTrainTest)

def preparePatchAB(freqMin, freqMax, decimation,frameSize,cardPatch,splitTrainTest,outputFormat):
    raise NotImplemented("Time needs time little hobbit")

#====== Feature manipulation (delete elec, step) =======
#=======================================================
class NotFittedError(Exception):pass

class CovScaler(object):
    def __init__(self):
        self.fitted = False

    def fit(self, X):
        cov = np.ma.cov(X)
        self.invCov = np.sqrt(np.linalg.inv(cov + 10e-2*np.eye(np.size(cov,0))))
        self.fitted = True

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        if not self.fitted:
            raise NotFittedError("Must call CovScaler.fit before CovScaler.transform")
        else:
            return np.dot(X.T,self.invCov).T


class Winsorizer(object):
    def __init__(self,cardElec):
        self.fitted = False
        self.limit = 0.001
        self.maximum = None
        self.minimum = None

        self.cardElec = cardElec


    def fit(self, X):
        raise NotImplemented("Should not be called alone")
    
    def fit_transform(self, X):
        self.fitted = True

        sizeSig = np.size(X,1)/self.cardElec

        for numElec in range(self.cardElec):
            elecSlice = slice(numElec*sizeSig,(numElec+1)*sizeSig,1)
            test = X[:,elecSlice]
            X[:,elecSlice] = winsorize(X[:,elecSlice],limits=self.limit)
            test = X[:,elecSlice]
            
        self.maximum = np.max(X)
        self.minimum = np.min(X)
        
        return X

    def transform(self, X):
        if not self.fitted:
            raise NotFittedError("Must call Winsorizer.fit_transform before Winsorizer.transform")
        else:
            X = np.clip(X,self.minimum,self.maximum)
            return X

    

def normalizeData(X,xTest,cardElec,scaleType='standard'):

    winSor = False
    if winSor:
        print "Winsorizing Data"
        winSor = Winsorizer(cardElec)
        X = winSor.fit_transform(X)
        xTest = winSor.transform(xTest)

    if scaleType == 'cov':
        print "Scaling Data using Covariance Matrix"
        cardExemple = np.size(X,0)
        X = np.concatenate((X,xTest))
        scaler = CovScaler()
        X = scaler.fit_transform(X)

        X, xTest =  X[:cardExemple,:],X[cardExemple:,:]
        print X.shape, xTest.shape
        
    elif scaleType=='standard':
        print "Scaling Data"
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if np.size(xTest)!=0:
            xTest = scaler.transform(xTest)

    elif scaleType=='sample':
        print "Normalizing Data"
        scaler = Normalizer()
        X = scaler.fit_transform(X)
        if np.size(xTest)!=0:
            xTest = scaler.transform(xTest)

    else:
        print("Data will not be normalized/scaled")

    return X,xTest

def delTimeStep(X, timeStep, dataType):

    if 'stft' in dataType or 'Stft' in dataType :
        raise NotImplemented("To be continued")

    else:
        numCol = np.size(X,1)
        numPoints = numCol//64
        
        mask = np.ones(numCol, dtype=bool)

        if isinstance(timeStep, int) or isinstance(timeStep, np.int64):
            for i in range(64):
                mask[timeStep+numPoints*i] = False

        else:
            for i in range(64):
                for step in timeStep:
                    mask[step+numPoints*i] = False
        # print(mask)
        X = X[:, mask]


    return X

def delElec(X, elec, dataType=None):

    numCol = np.size(X,1)
    numPoints = numCol//64

    mask = np.ones(numCol, dtype=bool)

    if isinstance(elec, int) or isinstance(elec, np.int64):
        mask[elec*numPoints:elec*(numPoints)+numPoints] = False
    else:
        for step in elec:
             mask[step*numPoints:step*(numPoints)+numPoints] = False

    # print(mask)
    X = X[:, mask]
    return X    


#========== CONVERT (EXEMPLE,CHANNEL,TIME) <=> (EXEMPLE, FEATURES) ==========
#============================================================================

def matToVec(X,cardElec=64):

    cardExemple = np.size(X,0)
    cardStep = np.size(X,2)

    xVec = X.reshape((cardExemple,cardStep*cardElec))

    return xVec

def vecToMat(X,cardElec=64):

    print X.shape

    cardExemple = np.size(X,0)
    cardStep = np.size(X,1)//cardElec

    xMat = X.reshape((cardExemple,cardElec,cardStep))
    return xMat

#========================= DIMENSION REDUCING ===================
#================================================================
def selectElectrode(X,xTest):

    # electrodes = [34,11,51,62]
    #electrodes = [34,11,51,62,47,49,53,55]
    electrodes = [34,11,51,62,47,49,53,55,61,63,17,19,9,13,3,5]
    # electrodes = [34,11,51,62,47,49,53,55,61,63,17,19,9,13,3,5,41,30,1,32,26,22,15,57,59,21,42,36,38,28,24]

    electrodes = set(electrodes)
    wholeElec = set(range(1,65))

    toDel = list(wholeElec - electrodes) 
    X = delElec(X, toDel)
    xTest = delElec(xTest, toDel)

    return X,xTest


def dimensionReducePCA(X,xTest,n_components=100):

    print("PCA : reducing {} features".format(np.size(X,1))
)

    pca = PCA(n_components=np.size(X,1)//10)
    X = pca.fit_transform(X)
    
    print("True size of X : ", X.shape)

    if xTest != []:
        xTest = pca.transform(xTest)

    return X,xTest


def transformLDA(X,y,xTest):
    
    originalSize = np.size(X,1)
    print("Learning LDA \nProjecting {} features to 1 component".format(originalSize))
    priors = [0.5,0.5]

    clf = LinearDiscriminantAnalysis('svd', n_components=1,priors=priors)
    print(X.shape)
    X = clf.fit_transform(X,y)
    print("True size of X : ", X.shape)

    if xTest != []:
        xTest = clf.transform(xTest)
    return X,xTest

def fisherCriterionFeaturesSelection(X,y,xTest,n_components):

    cardExemple, originalCardFeature = X.shape
    print("Learning Fisher Criterion custom : \nProjecting {} features to {} components".format(originalCardFeature, n_components))
    priors = [0.5,0.5]

    approach = 'fast'

    treshold = 0.90

    clf = LinearDiscriminantAnalysis('svd', n_components=n_components,priors=priors)
    print(X.shape)
    clf.fit(X,y)

    coef = np.abs(clf.coef_[0])

    # Sorting the coef list and keeping the index
    bestCoef,indexCoef = zip(*sorted(zip(coef,[i for i in range(originalCardFeature)])))
    bestCoef,indexCoef = list(reversed(bestCoef)), list(reversed(indexCoef))
    # Easy First Approach : Returning all best coef 

    print(max(bestCoef),min(bestCoef),np.mean(bestCoef),np.std(bestCoef))
    
    if approach=='fast':
        keptFeatures = indexCoef[:n_components]
        #Warning : Can be very correlated
        return X[:,keptFeatures], xTest[:,indexCoef[:n_components]]

    # Second Method : Selec coef if it is not too correlated to another    
    selectedFeatures = [indexCoef[0]]
    currentIndex = 1

    while len(selectedFeatures) != n_components and currentIndex < originalCardFeature-1:
        prd =  np.array([np.inner(X[:,indexCoef[currentIndex]],X[:,selectedFeatures[i]])/cardExemple for i in range(len(selectedFeatures))])

        
        if (prd < treshold).all() :
            selectedFeatures.append(indexCoef[currentIndex])
        currentIndex +=1

    return X[:,selectedFeatures], xTest[:,selectedFeatures]

def cspReduce(X,xTest,y,n_components):

    if np.size(xTest)==0:
        raise NoTestError("A test file is needed for dimension reducing, otherwise, test would be biaised.\nAdd '-r 0.8' option when calling mainParams.py")


    #Reformat X for csp function
    X = vecToMat(X)
    print X.shape
    
    #Apply CSP
    csp = CSP(n_components=n_components)
    X = csp.fit_transform(X,y)


    xTest = vecToMat(xTest)
    
    xTest = csp.transform(xTest)
        
    return X,xTest
    
#==================Patches Manipulation ===================
#==========================================================
def patchProcess(subject, freqMin, freqMax, decimation,frameSize,cardPatch,splitTrainTest,outputFormat,operationStr):

    print "Patch Process :"

    if os.path.exists('{}{}patched{}{}.npy'.format(PATH_TO_DATA,subject,operationStr.title(), cardPatch)):
        print "File Exists : Loading ..."
        return np.load('{}{}patched{}{}.npy'.format(PATH_TO_DATA,subject,operationStr.title(),cardPatch))

    else:
        print("Extracting Patches")
        X = np.load(PATH_TO_DATA+'{}fullFiltered{}_{}_{}StftMatrix{}X.npy'.format(subject,freqMin,freqMax,decimation,frameSize))
        patcher = Patcher(X,subject,operationStr,cardPatch)
        X = patcher.patchFeatures()
        return X
                    
class Patcher(object):
    def __init__(self,X,subject,operationStr,cardPatch=2, elecWidth=2,freqWidth=1, winWidth=1):

        self.X = X
        dictOpe = {'mean':np.mean, 'sum':np.sum,'max':np.max,'maxFreqWin':self._MaxFreqPerWin}
        self.operationStr = operationStr
        self.operation = dictOpe[self.operationStr]
        self.cardPatch = cardPatch
        self.elecWidth = elecWidth
        self.freqWidth = freqWidth
        self.winWidth = winWidth
        self.subject = subject

    def _MaxFreqPerWin(self,patch):
        maxFreq = patch.sum(axis=0).argmax(axis=0)
        assert np.size(maxFreq)==self.winWidth*2+1
        return maxFreq

    def generate1Patch(self):

        cardExemple, cardElec, cardFreq, cardWin = self.X.shape
        randElec = np.random.randint(low=self.elecWidth, high=cardElec-self.elecWidth, size=1)
        randFreq = np.random.randint(low=self.freqWidth, high=cardFreq-self.freqWidth, size=1)
        randWin = np.random.randint(low=self.winWidth, high=cardWin-self.winWidth, size=1)

        slices = [tuple(CLOSEST_ELEC[randElec][:2*self.elecWidth+1]),\
              slice(randFreq-self.freqWidth,randFreq+self.freqWidth+1),\
                   slice(randWin-self.winWidth,randWin+self.winWidth+1)]

        return self.X[0][slices]

        
        
    def generateXPatched(self):
        """
        Input
        ======
        X is a matrix of exemple, all of them represented in time-frequencies
        X must be a 4-D matrix (#Exemple x #Electrodes x #Frequencies x #Windows)

        xxxWidth : size of the patch on the xxx dimension : (2*xxxWidth)+1
        Ex : elecWidth = 2, 2*2+1=5, 5 electrodes will be taken in the patch

        Yield (Didn't use 'return' because of memory problem)
        =====
        Yield X patched, exemple by exemple, in a list format
        Dimension : cardPatch x 2*elecWidth+1 x 2*freqWidth+1 x 2*winWidth+1
        """

        assert np.ndim(self.X) == 4
        cardExemple, cardElec, cardFreq, cardWin = self.X.shape
        assert 2*self.elecWidth<cardElec
        assert 2*self.freqWidth<cardFreq
        assert 2*self.winWidth<cardWin


        randElec = np.random.randint(low=self.elecWidth, high=cardElec-self.elecWidth, size=self.cardPatch)
        randFreq = np.random.randint(low=self.freqWidth, high=cardFreq-self.freqWidth, size=self.cardPatch)
        randWin = np.random.randint(low=self.winWidth, high=cardWin-self.winWidth, size=self.cardPatch)

        slices = [(tuple(CLOSEST_ELEC[randElec[i]][:2*self.elecWidth+1]),\
              slice(randFreq[i]-self.freqWidth,randFreq[i]+self.freqWidth+1),\
              slice(randWin[i]-self.winWidth,randWin[i]+self.winWidth+1)) for i in range(self.cardPatch)]

        for ex in xrange(cardExemple):
            yield [self.X[ex][slices[i]] for i in range(self.cardPatch)]

        # Memory overload :
        # return [[self.X[ex][slices[i]] for i in range(cardPatch)] for ex in xrange(cardExemple)]


    def patchFeatures(self,save=True):
        cardExemple = np.size(self.X,0)
        sizePatch = np.size(self.operation(self.generate1Patch()))
        newX = np.empty((cardExemple, self.cardPatch*sizePatch), np.float64)

        for numEx, ex in enumerate(self.generateXPatched()):
            newX[numEx] = np.array([self.operation(patch) for patch in ex]).flatten()

            if not numEx%1000:
                print('Exemple Transformed : {}/{}'.format(numEx, cardExemple))

        self.X=newX
        self.saveData()
        return self.X

    def saveData(self):
        np.save("{}{}patched{}{}".format(PATH_TO_DATA,self.subject,self.operationStr.title(), self.cardPatch),self.X)


#======================= 1-usage function ===========
#====================================================
def savePartOfData(filenameI, filenameO):
    """
    Function only for test purpose
    """
    data = sio.loadmat("{}{}".format(PATH_TO_DATA, filenameI))
    miniData = np.array(data['X'][0])
    miniData = miniData.transpose()
    y = [elem[0] for elem in data['y']]

    np.save(filenameO, miniData)
    np.save('{}fullY'.format(PATH_TO_DATA), y)

def balanceData(X,y):

    print y
    allNegIndex = np.where(y==-1)[0]
    allPosIndex = np.where(y==1)[0]
    
    cardNeg = np.size(allNegIndex)
    cardPos = np.size(allPosIndex)

    print "Cardinal of Positive :", cardPos, "\nCardinal of Negative", cardNeg
    
    fewerNegIndex = np.array(list(set(allNegIndex[np.random.randint(low=0,high=cardNeg,size=cardPos+60)])))

    keptIndex= np.concatenate((fewerNegIndex,allPosIndex))    
    X = X[keptIndex]
    y = y[keptIndex].T

    print "Balanced Data : ", X.shape

    return X,y

    
def saveBalancedMat(subject):

    data = sio.loadmat("{}Subject_{}_Train_reshapedCopy.mat".format(PATH_TO_DATA,subject))
    y = np.array([elem[0] for elem in data['y']])
    X = np.array(data['X'])

    print X.shape

    data = {'X':None,'y':None}
    print y[0]

    
    cardExemple = np.size(X,2)
    newData = np.empty([cardExemple, 64*160])

    for exemple in range(cardExemple):
        newData[exemple, :] = np.concatenate( [X[i,:,exemple] for i in range(64)])

    X,y = balanceData(newData,y)
    
    data['X'] = X
    data['y'] = y
    print 'Here', data['y'].shape
    
    sio.savemat("{}Subject_{}_Train_reshaped.mat".format(PATH_TO_DATA,subject),data)

def saveMarseilleMat(subject):

    data = sio.loadmat("{}Subject_{}_Train_reshapedCopy.mat".format(PATH_TO_DATA,subject))

    print data.keys()

    X = data['X']
    data['X'] = X.reshape(np.size(X,0),np.size(X,1)*np.size(X,2))
    data['y'] = data['y'][:,1]

    print data['X'].shape
    print data['y']

    sio.savemat("{}Subject_{}_Train_reshaped.mat".format(PATH_TO_DATA,subject),data)

def saveDisabledDataBase(subject):

    wholeX = []
    wholeY = []
    for session in [1,2,3,4]:
        for trial in [1,2,3,4,5,6]:
            
            data = sio.loadmat("{}Subject_{}_{}{}.mat".format(PATH_TO_DATA,subject,session,trial))


            y = np.array(data['y'])
            X = np.array(data['X'])

            data = {'X':None,'y':None}
            cardElec, cardStep, cardExemple = X.shape
            
            newData = np.empty([cardExemple, cardElec*cardStep])

            for numExemple in range(cardExemple):
                newData[numExemple, :] = np.concatenate( [X[i,:,numExemple] for i in range(cardElec)])

            if wholeX == []:
                wholeX = newData
                wholeY = y
            else:
                wholeX = np.concatenate((wholeX,newData))
                wholeY = np.concatenate((wholeY,y),axis=1)

            assert np.size(wholeX,1) == np.size(newData,1)
            assert np.size(wholeY,0) == 1

    wholeY = wholeY[0] #whole Y is [[1,1,1,1 ...]] and we need [1,1,1,1...]
    
    wholeX,wholeY = balanceData(wholeX,wholeY)
    data['X'] = wholeX
    data['y'] = wholeY

    sio.savemat("{}Subject_{}_Train_reshaped.mat".format(PATH_TO_DATA,subject),data)
