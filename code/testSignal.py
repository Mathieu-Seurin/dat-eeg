#!/usr/bin/env python
# -*- coding: utf-8 -*-

from signalManipulation import *
from manipulateData import *

import sys

f0 = 5
fs = 60
T = 2

if sys.argv[1] == 'sin':

    x = np.linspace(0, T, fs*T, fs)

    signal = np.array([])
    signal = sin(2*pi*f0*x) #5Hz
    signal += sin(2*pi*2*f0*x) #10Hz

    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 8.75Hz


    mySig = SignalHandler(signal, fs)
    mySig.multipleStftPlot(frameSize=0.3)

elif sys.argv[1] == 'p300':
    X, y, _, _ = prepareRaw('A')
    
    p300 = np.where(y==1)[0]
    
    signal = X[p300[0],:166]
    mySig = SignalHandler(signal, fs)
    mySig.multipleStftPlot(frameSize=0.07)


elif sys.argv[1] == 'filter':

    fs = 240
    f0 = 40
    x = np.linspace(0, T, fs*T, fs)
        
    signal = sin(2*pi*f0*x) #40Hz
    signal += sin(2*pi*2*f0*x) ##80Hz
    signal += sin(2*pi*0.5*f0*x) ##20Hz
    signal += sin(2*pi*0.025*f0*x) #10Hz
    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 70Hz

    mySig = SignalHandler(signal, fs)
    mySig.plotFiltered()
    plt.show()


    X, _, _, _ = prepareRaw('A')

    mySig = SignalHandler(X[0, 0:160], 240)
    mySig.plotFiltered()


    # mySig.plotFiltered()



elif sys.argv[1] == 'wavelets':

    raise NotImplementedYet("Need Package wavelets")

    x = np.linspace(0, T, fs*T, fs)

    signal = sin(2*pi*f0*x) #40Hz
    signal += sin(2*pi*2*f0*x) #80Hz
    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 70Hz

    # mySig = SingleElectrodeProcessor(numSession=10, numElec=10)
    # signal = mySig.mainSignal

    dt = 0.1

    wa = WaveletAnalysis(signal, dt=dt)

    # wavelet power spectrum
    power = wa.wavelet_power
    print(power.shape, power)
    # scales
    scales = wa.scales
    print(scales.shape, scales)
    # associated time vector
    t = wa.time

    # reconstruction of the original data
    rx = wa.reconstruction()

    fig, ax = plt.subplots()
    T, S = np.meshgrid(t, scales)
    ax.contourf(T, S, power, 100)
    ax.set_yscale('log')

plt.show()
