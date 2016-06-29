#!/usr/bin/python3
#coding: utf-8

#Module
import pickle
import numpy as np
import scipy.io as sio
from scipy.sparse import lil_matrix
from signalManipulation import *
from sklearn.preprocessing import StandardScaler

import os.path
import time

class NotImplementedYet(Exception): pass

#Const
PATH_TO_DATA = 'Data/'
PATH_TO_MODEL = 'Models/'


class NotImplemented(Exception): pass

#======================= TEST ===========
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
    y = [elem[0] for elem in data['y']]
    data = np.array(data['X'])

    #TODO MODIFY 160
    numExemple = np.size(data,2)
    newData = np.empty([numExemple, 64*160])

    for exemple in range(numExemple):
        newData[exemple, :] = np.concatenate( [data[i,:,exemple] for i in range(64)])

    np.save('{}{}fullY'.format(PATH_TO_DATA, filenameO[0]), y)

    np.save('{}{}'.format(PATH_TO_DATA, filenameO), newData)

def filterRawData(filenameI, freqInf, freqSup, decimation):

    if filenameI[1:5] == 'full':
        subject = filenameI[0]
        print("Subject : ",subject)
    else:
        print(filenameI)
        raise NotImplemented("Can't use this format at the moment")

    #==== If File Already exist, don't recalculate everything : ====
    if os.path.exists(PATH_TO_DATA+subject+"fullFiltered{}_{}_{}RawX.npy".format(freqInf,freqSup,decimation)):
        print("filter Data : File already exists, loading ...")
        return 0
    #===============================================
    
    print("Filter Data : File doesn't exist, creating ...")


    data = np.load(PATH_TO_DATA+filenameI)
    numExemple = np.size(data,0)
    
    numPoints = np.size(data,1)//64
    print('Number of Points per signal : ',numPoints)

    currentSignal = SignalHandler(data[0,:numPoints], 240)
    
    sizeOfNewSignal = np.size(currentSignal.filterSig(6,freqInf,freqSup,decimation),0)
    print('Size of new signal : ', sizeOfNewSignal)

    newData = np.empty((numExemple, 64*sizeOfNewSignal))

    for ex, x in enumerate(data):
        for numSignal, i in enumerate(range(0,len(x), numPoints)):
            currentSignal.mainSignal = x[i:i+numPoints]
            signalFiltered = currentSignal.filterSig(6,freqInf,freqSup,decimation)

            indexBegin = numSignal*sizeOfNewSignal
            indexEnd = (numSignal+1)*sizeOfNewSignal
            newData[ex,indexBegin:indexEnd] = signalFiltered

        if not ex%1000:
            print('Transformed : {}/{}'.format(ex, numExemple))
    print(newData.shape)
    np.save(PATH_TO_DATA+subject+"fullFiltered{}_{}_{}RawX".format(freqInf,freqSup,decimation), newData)

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

    numPoints = np.size(data,1)//64
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

    
    newData = np.empty((numExemple, 64*sizeOfNewSignal))
    for ex, x in enumerate(data):
        #Since data is a concatenation of 64 electrode, you have to separate them
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

    dataType can be either 'raw' or 'filteredXX' with XX the order of filtering

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

def splitXY(fileX, fileY, dataTransformation,split=0.70):

    X = np.load(PATH_TO_DATA+fileX)
    y = np.load(PATH_TO_DATA+fileY)

    if split==0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X,y,np.array([]),np.array([])
    
    numExemples = np.size(X,0)
    print("""Splitting Data : \n Train : {:3.0f}%   Test : {:3.0f}%
    Total : {} Train : {:10.0f} Test : {:3.0f}""".format(split*100, (1-split)*100, numExemples, \
                                              numExemples*split, numExemples*(1-split))) 

    randSelection = np.random.sample(np.size(X,0))
    trainIndex = np.where(randSelection < split)
    testIndex = np.where(randSelection >= split)

    scaler = StandardScaler()
    
    xTrain = scaler.fit_transform(X[trainIndex])
    xTest = scaler.transform(X[testIndex])
    
    saveSplitted(xTrain, y[trainIndex], xTest, y[testIndex])
    return xTrain, y[trainIndex], xTest, y[testIndex]

def concatAB(fileXA, fileYA, fileXB, fileYB, dataType):

    xA, xB = np.load(PATH_TO_DATA+fileXA), np.load(PATH_TO_DATA+fileXB)
    yA, yB = np.load(PATH_TO_DATA+fileYA), np.load(PATH_TO_DATA+fileYB)

    fullX = np.concatenate((xA, xB))
    fullY = np.concatenate((yA, yB))

    np.save(PATH_TO_DATA+"ABfull{}X".format(dataType.title()), fullX)
    np.save(PATH_TO_DATA+"ABfullY", fullY)
    
def loadSplitted():
        
    X = np.load('{}{}'.format(PATH_TO_DATA, 'trainX.npy'))
    y = np.load('{}{}'.format(PATH_TO_DATA, 'trainY.npy'))
    xTest = np.load('{}{}'.format(PATH_TO_DATA, 'testX.npy'))
    yTest = np.load('{}{}'.format(PATH_TO_DATA, 'testY.npy'))

    return X,y,xTest,yTest

def prepareFiltered(subject, freqMin, freqMax, decimation, splitTrainTest=0):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject),"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    return splitXY("{}fullFiltered{}_{}_{}RawX.npy".format(subject,freqMin,freqMax,decimation),\
            "{}fullY.npy".format(subject), 'Filtered',splitTrainTest)

def prepareFilteredAB(freqMin,freqMax,decimation,splitTrainTest=0):

    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    filterRawData("AfullRawX.npy", freqMin, freqMax, decimation)

    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    filterRawData("BfullRawX.npy", freqMin, freqMax, decimation)

    concatAB("AfullFiltered{}_{}_{}RawX.npy".format(freqMin,freqMax,decimation),"AfullY.npy",\
             "BfullFiltered{}_{}_{}RawX.npy".format(freqMin,freqMax,decimation),"BfullY.npy",'Filtered')

    return splitXY("ABfullFilteredX.npy","ABfullY.npy", 'Filtered',splitTrainTest)
    
def prepareRaw(subject,splitTrainTest=0):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    return splitXY("{}fullRawX.npy".format(subject),"{}fullY.npy".format(subject) ,\
                   'Raw',splitTrainTest)

def prepareRawAB(splitTrainTest=0):
        
    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    concatAB("AfullRawX.npy","AfullY.npy", "BfullRawX.npy","BfullY.npy")

    return splitXY("ABfullRawX.npy","ABfullY.npy", 'Raw',splitTrainTest)

                                                     
def prepareStft(subject, frameSize,splitTrainTest=0):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    reformatStftData("{}fullRawX.npy".format(subject), 'Raw', 240, frameSize=frameSize) 

    return splitXY("{}fullRawStft{}X.npy".format(subject, frameSize),"{}fullY.npy".format(subject),\
                   'Stft'+str(frameSize), splitTrainTest)



def prepareFilteredStft(subject, freqMin, freqMax, decimation,frameSize, splitTrainTest=0):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftData(
        "{}fullFiltered{}_{}_{}RawX.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}'.format(decimation), 240//decimation, frameSize) 

    return splitXY("{}fullFiltered{}Stft{}X.npy".format(subject, decimation, frameSize),\
                   "{}fullY.npy".format(subject), "Filtered{}".format(decimation), splitTrainTest)

    

def prepareTransfertFiltered(freqMin, freqMax, decimation):

    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    filterRawData("AfullRawX.npy", freqMin, freqMax, decimation)

    X = np.load('{}AfullFiltered{}_{}_{}RawX.npy'.format(PATH_TO_DATA,freqMin, freqMax, decimation))
    y = np.load('{}{}'.format(PATH_TO_DATA, 'AfullY.npy'))

    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    filterRawData("BfullRawX.npy", freqMin, freqMax, decimation)

    xTest = np.load('{}BfullFiltered{}_{}_{}RawX.npy'.format(PATH_TO_DATA,freqMin, freqMax, decimation))
    yTest = np.load('{}{}'.format(PATH_TO_DATA, 'BfullY.npy'))

    scaler = StandardScaler()
    
    X = scaler.fit_transform(X)
    xTest = scaler.transform(xTest)
    
    return X, y, xTest, yTest

#====== Feature manipulation (delete elec, step) =======
#=======================================================
def delTimeStep(X, timeStep, dataType):

    if 'stft' in dataType or 'Stft' in dataType :
        raise NotImplementedYet("To be continued")

    else:
        numEx = np.size(X,0)
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

def delElec(X, elec, dataType):

    numEx = np.size(X,0)
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


################### Patches Manipulation ##############
class Patcher(object):
    def __init__(self,X,operationStr,cardPatch=2, elecWidth=2,freqWidth=1, winWidth=1):

        self.X = X
        dictOpe = {'mean':np.mean, 'sum':np.sum}
        self.operationStr = operationStr
        self.operation = dictOpe[self.operationStr]
        self.cardPatch = cardPatch
        self.elecWidth = elecWidth
        self.freqWidth = freqWidth
        self.winWidth = winWidth
        
    def generateXPatched(self):
        """
        Idée : Tenir compte de la topologie des électrodes pour la sélection

        Input
        ======
        X is a matrix of exemple, all of them represented in time-frequencies
        X must be a 4-D matrix (#Exemple x #Electrodes x #Frequencies x #Windows)

        xxxWidth : size of the patch on the xxx dimension : (2*xxxWidth)+1
        Ex : elecWidth = 2, 2*2+1=5, 5 electrode will be taken in the patch

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

        slices = [(slice(randElec[i]-self.elecWidth,randElec[i]+self.elecWidth+1),\
              slice(randFreq[i]-self.freqWidth,randFreq[i]+self.freqWidth+1),\
              slice(randWin[i]-self.winWidth,randWin[i]+self.winWidth+1)) for i in range(self.cardPatch)]

        for ex in xrange(cardExemple):
            yield [self.X[ex][slices[i]] for i in range(self.cardPatch)]

        # Memory overload :
        # return [[self.X[ex][slices[i]] for i in range(cardPatch)] for ex in xrange(cardExemple)]


    def patchFeatures(self,save=True):
        cardExemple = np.size(self.X,0)
        newX = np.empty( (cardExemple, self.cardPatch), np.float64)

        for numEx, ex in enumerate(self.generateXPatched()):
            newX[numEx] = [self.operation(patch) for patch in ex]

        self.X=newX
        self.saveData()
        return self.X

    def saveData(self):
        np.save("{}patched{}{}".format(PATH_TO_DATA,self.operationStr.title(), self.cardPatch),self.X)
