#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rawManipulation import *
import sys

f0 = 40
fs = 240
T = 2

if sys.argv[1] == 'sin':

    x = np.linspace(0, T, fs*T, fs)

    signal = sin(2*pi*f0*x) #40Hz
    signal += sin(2*pi*2*f0*x) #80Hz


    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 70Hz
    mySig = SignalHandler(signal, fs)
    mySig.multiplePlot()

    # mySig.plot()
    # mySig.plotStft()

elif sys.argv[1] == 'elec':

    mySig = SingleElectrodeProcessor(numSession=10, numElec=10)
    mySig.multiplePlot()
    # mySig.plot()
    # mySig.plotFft()
    # mySig.plotStft()

elif sys.argv[1] == 'cut':

    mySig = MultipleElectrodeProcessor()
    signal = mySig.get(0,0,7)

    sample = SignalHandler(signal)
    sample.multiplePlot()


    # sample.plot()
    # sample.plotFft()
    # sample.plotStft()


plt.show()
