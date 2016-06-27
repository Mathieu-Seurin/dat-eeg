#!/usr/bin/env python
# -*- coding: utf-8 -*

from learnData import *
from manipulateData import *
import os
import sys

test = sys.argv[1]
frameSize = 0.2


if test == 'stftMatRaw':
    subject = 'A'
    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject))
    reformatStftDataMatrix("{}fullRawX.npy".format(subject), 'Raw', 240, frameSize=frameSize,outputFormat='npy') 
    
elif test == 'stftMatFilter':
    subject = 'A'
    freqMin = float(sys.argv[2])
    freqMax = float(sys.argv[3])
    decimation = int(sys.argv[4])

    reformatRawData("Subject_{}_Train_reshaped.mat".format(subject) ,"{}fullRawX".format(subject))
    filterRawData("{}fullRawX.npy".format(subject), freqMin, freqMax, decimation)

    reformatStftDataMatrix(
        "{}fullFiltered{}_{}_{}RawX.npy".format(subject, freqMin, freqMax, decimation),
        'Filtered{}'.format(decimation), 240//decimation, frameSize=frameSize, outputFormat='mat') 
