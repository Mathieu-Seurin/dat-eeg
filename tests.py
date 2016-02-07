#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matrixManipulation import *
import sys

f0 = 10
fs = 240
T = 1

if sys.argv[1] == 'sin':

    x = np.linspace(0, T, fs*T, fs)
    signal = sin(2*pi*f0*x)+sin(2*pi*(f0/2)*x)

    mySig = SignalHandler(signal, fs)
    mySig.multiplePlot()

    # mySig.plot()
    # mySig.plotFft()
    # mySig.plotStft()

elif sys.argv[1] == 'elec':

    mySig = SingleElectrodeProcessor(numSession=10, numElec=10)
    mySig.multiplePlot()
    # mySig.plot()
    # mySig.plotFft()
    # mySig.plotStft()

elif sys.argv[1] == 'cut':

    mySig = MultipleElectrodeProcessor()
    signal = mySig.get()

    sample = SignalHandler(signal)
    sample.multiplePlot()


    # sample.plot()
    # sample.plotFft()
    # sample.plotStft()
    # plt.show()
