#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matrixManipulation import *
import sys

f0 = 10
fs = 240
T = 1

if sys.argv[1] == 'sin':

    x = np.linspace(0, fs*T, fs)
    signal = sin(2*pi*f0*x)

    mySig = SignalHandler(signal, fs)
    mySig.plot()
    mySig.plotFft()
    mySig.plotStft(numWindow=10)

elif sys.argv[1] == 'elec':

    mySig = SingleElectrodeProcessor(numSession=10, numElec=10)
    mySig.plot()
    mySig.plotFft()
    mySig.plotStft(numWindow=2)

elif sys.argv[1] == 'cut':

    mySig = MultipleElectrodeProcessor()
    signal = mySig.get()

    print(len(signal))
    sample = SignalHandler(signal)
    sample.plot()
    sample.plotFft()
