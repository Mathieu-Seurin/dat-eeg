#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

test = sys.argv[1]
subject = 'A'
freqMin = float(sys.argv[2])
freqMax = float(sys.argv[3])
decimation = int(sys.argv[4])

frameSize = 0.2


if test == 'stftMatRaw':
    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject))
    reformatStftDataMatrix("{}fullRawX.npy".format(subject), 'Raw', 240, frameSize=frameSize,outputFormat='npy')

elif test == 'stft':
    prepareFilteredStft(subject,freqMin,freqMax,decimation,frameSize)
    
elif test == 'stftMatFilter':
    
    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftDataMatrix(
        "{}fullFiltered{}_{}_{}RawX.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}'.format(decimation), 240//decimation, frameSize=frameSize, outputFormat='npy')


    Xmat = np.load(PATH_TO_DATA+"{}fullFiltered{}StftMatrix{}X.npy".format(subject, decimation, frameSize))
    Xvec = np.load(PATH_TO_DATA+"{}fullFiltered{}Stft{}X.npy".format(subject, decimation, frameSize))
    t1m = Xmat[0]
    t1v = Xvec[0,:]

    print t1m
    print t1m.size
    
    print t1v
    print t1v.size

elif test == 'patch':
    X = np.load(PATH_TO_DATA+'AfullFiltered4StftMatrix0.2X.npy')
    patcher = Patcher(X,'mean',100)

    a = time.time()
    X = patcher.patchFeatures()
    print(X.shape)
    print(time.time()-a)
