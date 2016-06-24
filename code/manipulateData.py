#!/usr/bin/python3
#coding: utf-8

#Module
import pickle
import numpy as np
import scipy.io as sio

from signalManipulation import *
from sklearn.preprocessing import StandardScaler

import os.path

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

    #TODO MODIFY 160 line 136
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
    
    freqBeforeProcess = fftfreq(numFreqBeforeProcess*2, 1/fs)

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

def splitXY(fileX, fileY, dataTransformation,  split=0.70):

    X = np.load(PATH_TO_DATA+fileX)
    y = np.load(PATH_TO_DATA+fileY)

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


def delTimeStep(X, timeStep, dataType):

    if dataType!='stft' :
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
    else:
        raise NotImplementedYet("To be continued")

    return X

def delElec(X, elec, dataType):

    if dataType!='stft' :
        
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
    else:
        raise NotImplementedYet("To be continued")

    return X    
    
def loadSplitted():
        
    X = np.load('{}{}'.format(PATH_TO_DATA, 'trainX.npy'))
    y = np.load('{}{}'.format(PATH_TO_DATA, 'trainY.npy'))
    xTest = np.load('{}{}'.format(PATH_TO_DATA, 'testX.npy'))
    yTest = np.load('{}{}'.format(PATH_TO_DATA, 'testY.npy'))

    return X,y,xTest,yTest

def prepareFiltered(subject, freqMin, freqMax, decimation):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject),"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    splitXY("{}fullFiltered{}_{}_{}RawX.npy".format(subject,freqMin,freqMax,decimation),\
            "{}fullY.npy".format(subject), 'Filtered')

    return loadSplitted()

def prepareFilteredAB(freqMin,freqMax,decimation):

    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    filterRawData("AfullRawX.npy", freqMin, freqMax, decimation)

    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    filterRawData("BfullRawX.npy", freqMin, freqMax, decimation)

    concatAB("AfullFiltered{}_{}_{}RawX.npy".format(freqMin,freqMax,decimation),"AfullY.npy",\
             "BfullFiltered{}_{}_{}RawX.npy".format(freqMin,freqMax,decimation),"BfullY.npy",'Filtered')

    splitXY("ABfullFilteredX.npy","ABfullY.npy", 'Filtered')

    return loadSplitted()
    
def prepareRaw(subject):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    splitXY("{}fullRawX.npy".format(subject),"{}fullY.npy".format(subject) , 'Raw')
    
    return loadSplitted()

def prepareRawAB():
        
    reformatRawData("Subject_A_Train_reshaped.mat","AfullRawX")
    reformatRawData("Subject_B_Train_reshaped.mat","BfullRawX")
    concatAB("AfullRawX.npy","AfullY.npy", "BfullRawX.npy","BfullY.npy")

    splitXY("ABfullRawX.npy","ABfullY.npy", 'Raw')
    
    return loadSplitted()

                                                     
def prepareStft(subject, frameSize):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    reformatStftData("{}fullRawX.npy".format(subject), 'Raw', 240, frameSize=frameSize) 
    splitXY("{}fullRawStft{}X.npy".format(subject, frameSize),"{}fullY.npy".format(subject), 'Stft'+str(frameSize))



def prepareFilteredStft(subject, freqMin, freqMax, decimation,frameSize):

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject) )
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftData(
        "{}fullFiltered{}_{}_{}RawX.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}'.format(decimation), 240//decimation, frameSize) 

    splitXY("{}fullFiltered{}Stft{}X.npy".format(subject, decimation, frameSize),"{}fullY.npy".format(subject), "Filtered{}".format(decimation))

    
    return loadSplitted()

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

#==============================================================================
