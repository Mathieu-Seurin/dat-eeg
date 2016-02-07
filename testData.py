#!/usr/bin/env python
# -*- coding: utf-8 -*

from matrixManipulation import *

def signalCutting(signal):
    """
    Function to segment the signal into trial (instead of whole session)
    Input : One session of 7794 points (32,45s)
    Ouput : List of Trial (504 or 522 points)

    One session look like this :

    12*175 ms : One sample (set of 12 intensification) 2100 ms : 504 Points

    15*504 points : 15 Samples

    7560 points at 240Hz
    """

    oldSignal = signal
    newSignal = []

    for trial in range(15):
        newSignal.append( oldSignal[trial*504:(trial+1)*504] )

    assert len(newSignal)==15
    return newSignal


data = sio.loadmat("BCI/Subject_A_Train.mat")

dataR = dict((letter, []) for letter in data['TargetChar'][0])
dataR['absent'] = []

print(dataR)

trial = data['StimulusType'][1]
print(trial[:42])
print(trial[42:84])


code = data['StimulusCode'][0]
numW = np.where(code)
print(np.size(numW))
print(numW[0])

# for numSession, session in enumerate(data['Signal']):
#     print(data['TargetChar'][0][numSession])
#     print(len(data['StimulusType'][numSession]))
#
#     cutSignal(session)
#     input()
